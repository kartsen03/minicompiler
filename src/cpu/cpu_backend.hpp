#pragma once

#include "minicompiler/runtime/backend.hpp"

#include <memory>

namespace minicompiler {

std::unique_ptr<Backend> make_cpu_backend();

}
