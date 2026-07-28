# Bandicoot caches compiled OpenCL program binaries on disk:
#   $HOME/.bandicoot/cache/<kernel>/<host_device_id>   on POSIX
#   %APPDATA%\bandicoot\cache\...                      on Windows
# (inst/include/bandicoot_bits/cache_meat.hpp).
#
# COOT_KERNEL_CACHE_DIR is a compile-time #define, not an environment
# variable, and src/Makevars never sets it -- so redirecting HOME and APPDATA
# is the only way to stop R CMD check writing outside the session temp
# directory.  Both are set: Windows is the leg most likely to have a device,
# and it is the one that uses APPDATA.
local({
  cache <- file.path(tempdir(), "bandicoot-cache")
  dir.create(cache, showWarnings = FALSE, recursive = TRUE)
  Sys.setenv(HOME = cache, APPDATA = cache)
})
