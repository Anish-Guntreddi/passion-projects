#include "test_socket_client.hpp"

#include <algorithm>
#include <charconv>
#include <vector>

#include "arcserve/net/socket.hpp"

namespace arcserve::testing {

RawTcpClient::RawTcpClient(std::uint16_t port) : fd_(net::connect_loopback(port)) {}

bool RawTcpClient::send_fragmented(std::string_view data, std::size_t chunk_size) {
  if (data.empty()) {
    return true;
  }
  if (chunk_size == 0) {
    chunk_size = data.size();
  }

  std::size_t offset = 0;
  while (offset < data.size()) {
    std::size_t n = std::min(chunk_size, data.size() - offset);
    net::IoResult result = net::write_all(fd_.get(), data.data() + offset, n);
    if (result.status != net::IoStatus::Ok) {
      return false;
    }
    offset += n;
  }
  return true;
}

std::string RawTcpClient::read_chunk(std::size_t max_bytes) {
  std::vector<char> buffer(max_bytes);
  net::IoResult result = net::read_some(fd_.get(), buffer.data(), buffer.size());
  if (result.status == net::IoStatus::Closed || result.status == net::IoStatus::Error) {
    closed_ = true;
  }
  return std::string(buffer.data(), result.bytes_transferred);
}

void RawTcpClient::close_now() {
  fd_.reset();
  closed_ = true;
}

std::string read_one_http_response(RawTcpClient& client) {
  static const std::string kTerminator = "\r\n\r\n";
  std::string buffer;

  std::size_t header_end = std::string::npos;
  while (header_end == std::string::npos) {
    std::string chunk = client.read_chunk();
    if (chunk.empty() && client.is_closed()) {
      return buffer;
    }
    buffer += chunk;
    header_end = buffer.find(kTerminator);
  }

  std::size_t content_length = 0;
  {
    std::string headers = buffer.substr(0, header_end);
    static const std::string kKey = "Content-Length:";
    auto pos = headers.find(kKey);
    if (pos != std::string::npos) {
      pos += kKey.size();
      while (pos < headers.size() && headers[pos] == ' ') {
        ++pos;
      }
      std::size_t end = headers.find("\r\n", pos);
      if (end == std::string::npos) {
        end = headers.size();
      }
      std::from_chars(headers.data() + pos, headers.data() + end, content_length);
    }
  }

  std::size_t needed = header_end + kTerminator.size() + content_length;
  while (buffer.size() < needed) {
    std::string chunk = client.read_chunk();
    if (chunk.empty() && client.is_closed()) {
      break;
    }
    buffer += chunk;
  }

  // Guard against accidentally having read the start of a *second*
  // pipelined response in the same recv() — not expected given how the
  // tests drive this client (one request in flight at a time), but cheap
  // to make robust rather than assumed.
  if (buffer.size() > needed) {
    buffer.resize(needed);
  }
  return buffer;
}

}  // namespace arcserve::testing
