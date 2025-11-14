// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2022 Ryan Curtin (https://www.ratml.org/)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

// If in write mode, we will create the file (and directories if needed).
inline
std::fstream
open_cache(const std::string& unique_host_device_id,
           const std::string& kernel_name,
           const bool write)
  {
  // Strategy:
  //
  //    - search COOT_KERNEL_CACHE_DIR (if defined)
  //    - search COOT_SYSTEM_KERNEL_CACHE_DIR (defaults to /var/cache/bandicoot/)
  //    - search $HOME/.bandicoot/cache/
  //

  std::fstream f;

  // First check COOT_KERNEL_CACHE_DIR, if it's defined.
  #if defined(COOT_KERNEL_CACHE_DIR)
    {
    try_open(f, COOT_KERNEL_CACHE_DIR + std::string("/") + kernel_name + std::string("/"), unique_host_device_id, write);
    if (f.is_open() && f.good())
      return f;

    coot_extra_debug_warn(std::string("could not open COOT_KERNEL_CACHE_DIR: ") + std::string(COOT_KERNEL_CACHE_DIR) + std::string(", moving on"));
    }
  #endif

  // Next check COOT_SYSTEM_KERNEL_CACHE_DIR, if it's defined.
  #if defined(COOT_SYSTEM_KERNEL_CACHE_DIR)
    {
    try_open(f, COOT_SYSTEM_KERNEL_CACHE_DIR + std::string("/") + kernel_name + std::string("/"), unique_host_device_id, write);
    if (f.is_open() && f.good())
      return f;

    coot_extra_debug_warn(std::string("could not open COOT_SYSTEM_KERNEL_CACHE_DIR: ") + std::string(COOT_SYSTEM_KERNEL_CACHE_DIR) + std::string(", moving on"));
    }
  #endif

  #if defined(COOT_HAS_POSIX_FILE_FUNCTIONS)
    {
    // We are on Linux or OS X (or something exotic).
    const char* homedir = getenv("HOME");
    if (homedir == NULL)
      {
      homedir = getpwuid(getuid())->pw_dir;
      }

    try_open(f, std::string(homedir) + std::string("/.bandicoot/cache/") + kernel_name + std::string("/"), unique_host_device_id, write);
    }
  #elif defined(WIN32)
    {
    // We are on Windows.
    const char* user_profile_dir = getenv("APPDATA");
    try_open(f, std::string(user_profile_dir) + std::string("\\bandicoot\\cache\\") + kernel_name + std::string("\\"), unique_host_device_id, write);
    }
  #else
    {
    // No implementation available!
    coot_debug_warn("could not find home directory on this system to store kernel cache");
    }
  #endif

  return f;
  }



inline
void
try_open(std::fstream& f,
         const std::string& dirname,
         const std::string& filename,
         const bool write)
  {
  #ifdef COOT_HAS_POSIX_FILE_FUNCTIONS
    {
    struct stat info;
    if (stat(dirname.c_str(), &info) == -1)
      {
      // Check to see what the error was.
      // If the directory simply doesn't exist, we can try and make it if `write` is true.
      if ((errno == ENOENT || errno == ENOTDIR) && write)
        {
        if (!try_recursive_mkdir(dirname))
          {
          coot_extra_debug_warn(std::string("failed to create directory ") + dirname + std::string("' for kernel cache"));
          return; // We failed to make the directory, so we can't open the stream.
          }
        else
          {
          if (stat(dirname.c_str(), &info) == -1)
            {
            // Something else is wrong.
            return;
            }
          }
        }
      else
        {
        coot_extra_debug_warn(std::string("error while opening kernel cache directory '") + dirname + std::string("': ")  + std::string(strerror(errno)));
        return; // No stream can be opened.
        }
      }

    if (!(info.st_mode & S_IFDIR))
      {
      // The directory is not a directory, so we can't do anything.
      coot_extra_debug_warn(std::string("cache directory '") + dirname + std::string("' is not a directory"));
      return;
      }

    // Now attempt to open the file.  Processing whether it succeeds is up to the caller.
    f.open(dirname + filename,
           write ? (std::fstream::binary | std::fstream::trunc | std::fstream::out)
                 : (std::fstream::binary | std::fstream::in));
    if (!f.is_open())
      {
      coot_extra_debug_warn(std::string("opening kernel cache '") + dirname + filename + std::string("' failed: ") + std::string(strerror(errno)));
      }
    }
  #endif
  }



inline
bool
try_recursive_mkdir(const std::string& dirname)
  {
  #if defined(COOT_HAS_POSIX_FILE_FUNCTIONS)
    {
    // The directory delimiter is expected to be the last character.
    char delim = dirname[dirname.size() - 1];

    char tmp[PATH_MAX];
    snprintf(tmp, sizeof(tmp), "%s", dirname.c_str());

    // We'll just try from the root of the filesystem.  If we get an EEXIST, that directory already exists and we can just keep moving.
    for (size_t i = 1; i < dirname.size(); ++i)
      {
      if (tmp[i] == delim)
        {
        tmp[i] = '\0';
        if (!mkdir(tmp, S_IRWXU))
          {
          // Ignore any EEXISTs.  If we got something else, we fail.
          if (errno != EEXIST)
            return false;
          }
        tmp[i] = delim;
        }
      }

    return true;
    }
  #else
    {
    return false;
    }
  #endif
  }



inline size_t has_cached_kernel(const std::string& unique_host_device_id,
                                const std::string& kernel_name)
  {
  coot_extra_debug_sigprint();

  std::fstream f = open_cache(unique_host_device_id, kernel_name, false);

  // If opening the file failed for any reason, reject.
  if (!f.is_open() || !f.good())
    {
    coot_extra_debug_warn(std::string("failed to open kernel cache for device ") + unique_host_device_id + std::string(" and kernel ") + kernel_name);
    return 0;
    }

  // Make sure the stored version number matches the current version number.
  size_t f_ver_major, f_ver_minor, f_ver_patch;
  f.read((char*) &f_ver_major, sizeof(size_t));
  f.read((char*) &f_ver_minor, sizeof(size_t));
  f.read((char*) &f_ver_patch, sizeof(size_t));
  if (f_ver_major != COOT_VERSION_MAJOR || f_ver_minor != COOT_VERSION_MINOR || f_ver_patch != COOT_VERSION_PATCH)
    {
    std::ostringstream oss;
    oss << "kernel cache for kernel '" << kernel_name << "' was created by incorrect bandicoot version " << f_ver_major << "." << f_ver_minor << "." << f_ver_patch;
    coot_extra_debug_warn(oss.str());
    return 0;
    }

  if (!f.good())
    {
    coot_warn(std::string("error reading bandicoot version from kernel cache for kernel '") + kernel_name + std::string("': ") + std::string(strerror(errno)));
    return 0;
    }

  // Now get the size of the cached kernels.
  size_t f_kernel_size;
  f.read((char*) &f_kernel_size, sizeof(size_t));
  if (!f.good())
    {
    coot_warn(std::string("error reading kernel size from kernel cache for kernel '") + kernel_name + std::string("': ") + std::string(strerror(errno)));
    return 0;
    }

  return f_kernel_size;
  }



// function to get compiled program
// this assumes memory is allocated

inline bool read_cached_kernel(const std::string& unique_host_device_id,
                               const std::string& kernel_name,
                               unsigned char* buffer)
  {
  coot_extra_debug_sigprint();

  std::fstream f = open_cache(unique_host_device_id, kernel_name, false);

  // If opening the file failed for some reason, fail.
  if (!f.is_open() || !f.good())
    return false;

  // Skip the version, get the size.
  size_t s;
  f.read((char*) &s, sizeof(size_t));
  f.read((char*) &s, sizeof(size_t));
  f.read((char*) &s, sizeof(size_t));
  f.read((char*) &s, sizeof(size_t));
  if (!f.good())
    {
    coot_warn(std::string("error reading from kernel cache: ") + std::string(strerror(errno)));
    return false;
    }

  // Now read into the buffer.
  f.read((char*) buffer, s);
  return f.good();
  }



inline bool cache_kernel(const std::string& unique_host_device_id,
                         const std::string& kernel_name,
                         const unsigned char* buffer,
                         const size_t buf_len)
  {
  coot_extra_debug_sigprint();

  std::fstream f = open_cache(unique_host_device_id, kernel_name, true);

  // If opening the file failed for some reason, fail.
  if (!f.is_open() || !f.good())
    {
    coot_warn(std::string("failed to open kernel cache for writing for device ") + unique_host_device_id + std::string(" and kernel ") + kernel_name);
    return false;
    }

  size_t f_v_major, f_v_minor, f_v_patch;
  f_v_major = size_t(COOT_VERSION_MAJOR);
  f_v_minor = size_t(COOT_VERSION_MINOR);
  f_v_patch = size_t(COOT_VERSION_PATCH);

  // Now output the current version number of Bandicoot, followed by the size, followed by the kernels.
  f.write((char*) &f_v_major, sizeof(size_t));
  f.write((char*) &f_v_minor, sizeof(size_t));
  f.write((char*) &f_v_patch, sizeof(size_t));
  f.write((char*) &buf_len, sizeof(size_t));
  f.write((char*) buffer, buf_len);
  if (!f.good())
    {
    coot_warn(std::string("error writing kernel cache: ") + std::string(strerror(errno)));
    return false;
    }

  return true;
  }
