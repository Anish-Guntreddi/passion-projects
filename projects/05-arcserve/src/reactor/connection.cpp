#include "arcserve/reactor/connection.hpp"

#include "arcserve/net/socket.hpp"

namespace arcserve::reactor {

FlushResult Connection::flush_output() noexcept {
  while (!write_queue_.empty()) {
    std::string_view chunk = write_queue_.front();
    net::IoResult result = net::write_all(fd_.get(), chunk.data(), chunk.size());
    if (result.bytes_transferred > 0) {
      write_queue_.consume(result.bytes_transferred);
      touch();
    }
    if (result.status == net::IoStatus::Closed || result.status == net::IoStatus::Error) {
      return FlushResult::kFailed;
    }
    if (result.status == net::IoStatus::WouldBlock) {
      return FlushResult::kPending;
    }
    // Ok: net::write_all() only ever returns Ok once the entire requested
    // length was sent (it loops internally over partial sends/EINTR until
    // then), so the front chunk is now fully drained — loop to the next
    // one, if any.
  }
  return FlushResult::kFlushed;
}

}  // namespace arcserve::reactor
