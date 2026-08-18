#include "pebbledb/manifest/manifest.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>

#include "pebbledb/util/posix_file.hpp"

namespace pebbledb::manifest {

namespace {

std::string ManifestPath(const std::string& dbname) { return dbname + "/" + kManifestFileName; }
std::string TempManifestPath(const std::string& dbname) {
  return dbname + "/" + kManifestTempFileName;
}

}  // namespace

Status LoadManifest(const std::string& dbname, ManifestData* out) {
  const std::string path = ManifestPath(dbname);

  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    return Status::NotFound("no manifest at '" + path + "'");
  }

  std::unique_ptr<util::PosixFile> file;
  Status s = util::PosixFile::OpenRead(path, &file);
  if (!s.ok()) {
    return s;
  }

  std::uint64_t size = 0;
  s = file->Size(&size);
  if (!s.ok()) {
    return s;
  }

  std::string contents;
  bool short_read = false;
  s = file->Read(static_cast<std::size_t>(size), &contents, &short_read);
  if (!s.ok()) {
    return s;
  }
  // A short_read here would mean the file shrank between exists()/Size()
  // and this Read() -- an external race this project's single-writer
  // model does not expect to happen in practice. It is not specially
  // handled: DecodeManifest below simply sees fewer bytes than expected
  // and reports kTruncated, which is folded into Status::Corruption the
  // same as any other malformed manifest.

  DecodeStatus ds = DecodeManifest(contents, out);
  if (ds != DecodeStatus::kOk) {
    return Status::Corruption(std::string("manifest decode failed: ") + DecodeStatusName(ds));
  }
  return Status::OK();
}

Status SaveManifest(const std::string& dbname, const ManifestData& data, bool* published) {
  if (published != nullptr) {
    *published = false;
  }

  const std::string temp_path = TempManifestPath(dbname);
  const std::string final_path = ManifestPath(dbname);

  std::string encoded;
  EncodeManifest(data, &encoded);

  {
    std::unique_ptr<util::PosixFile> file;
    Status s = util::PosixFile::OpenTruncate(temp_path, &file);
    if (!s.ok()) {
      return s;
    }
    s = file->Append(encoded);
    if (!s.ok()) {
      return s;
    }
    s = file->Sync();
    if (!s.ok()) {
      return s;
    }
    // `file` (and its fd) closes here, at the end of this scope, before
    // the rename below -- not strictly required for correctness on
    // POSIX (a rename doesn't care whether the source path is still
    // open), but there is no reason to hold it open any longer than
    // needed.
  }

  if (std::rename(temp_path.c_str(), final_path.c_str()) != 0) {
    return Status::IOError("rename('" + temp_path + "' -> '" + final_path +
                            "') failed: " + std::strerror(errno));
  }

  // The rename above is the actual publish point (step 3 in manifest.hpp's
  // doc comment) -- `data` is now durably-on-this-filesystem's-view the
  // content of `final_path`, regardless of what happens next. Record that
  // *before* attempting the directory fsync below, so a caller can tell
  // "rename never happened" (published stays false, see the two failure
  // shapes documented on manifest.hpp's SaveManifest()) apart from "rename
  // happened, only the trailing directory-fsync failed" (ADR 0016) --
  // treating those two identically is what let a prior version of
  // DB::FlushLocked() keep serving out of stale in-memory state that had
  // already been superseded on disk.
  if (published != nullptr) {
    *published = true;
  }

  // This fsync makes the rename's directory-entry mutation itself durable
  // across a crash, mirroring util::PosixFile's own "fsync the directory
  // after creating/replacing a directory entry" rule (docs/durability.md).
  // A failure here does NOT undo the rename above -- the manifest file
  // itself already has the new content -- so `published` stays true.
  return util::SyncDirectory(dbname);
}

}  // namespace pebbledb::manifest
