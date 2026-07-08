#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/types/span.h"
#include "third_party/neptune/data/processor/cc/api/registry.h"
#include "third_party/neptune/data/processor/cc/core/transform.h"
#include "third_party/neptune/data/processor/cc/core/transform_registry.h"
#include "third_party/neptune/data/processor/proto/transform_config.proto.h"
#include "third_party/neptune/data/processor/python/examples/int_resource.h"
#include "third_party/neptune/data/resource_manager/cc/resource.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "tensorflow/compiler/xla/python/ifrt/dtype.h"

namespace jax_data::composite {

using ::neptune::data::composite::IntResource;
using ::neptune::data::composite::TransformBase;
using ::neptune::data::composite::TransformConfig;

class CustomTestTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<CustomTestTransform>> Create(
      std::shared_ptr<const IntResource> resource) {
    return std::make_unique<CustomTestTransform>(std::move(resource));
  }

  explicit CustomTestTransform(std::shared_ptr<const IntResource> resource)
      : resource_(std::move(resource)) {}

  absl::StatusOr<jsv::HostBuffer> operator()(jsv::HostBuffer a) {
    if (resource_ == nullptr) {
      return absl::FailedPreconditionError("Resource not loaded.");
    }
    if (a.dtype().kind() != xla::ifrt::DType::kS32) {
      return absl::InvalidArgumentError("Input buffer must be S32");
    }
    jsv::HostBuffer out_buffer = jsv::HostBuffer(a.dtype(), a.shape());
    int64_t num_elements = a.shape().num_elements();
    const int* data = static_cast<const int*>(a.base());
    int* out_data = static_cast<int*>(out_buffer.base());
    int res_val = resource_->value();
    for (int64_t i = 0; i < num_elements; ++i) {
      out_data[i] = data[i] + res_val;
    }
    return out_buffer;
  }

  absl::StatusOr<TransformConfig> ToProto() const override {
    TransformConfig config;
    config.set_op_name("CustomTestTransform");
    if (resource_ != nullptr) {
      config.add_resource_ids(resource_->id());
    }
    return config;
  }

  std::vector<std::shared_ptr<neptune::data::Resource>> GetAllResources()
      const override {
    if (resource_ == nullptr) {
      return {};
    }
    return {std::static_pointer_cast<neptune::data::Resource>(
        std::const_pointer_cast<IntResource>(resource_))};
  }

  absl::string_view DebugString() const override {
    return "CustomTestTransform";
  }

 private:
  std::shared_ptr<const IntResource> resource_;
};

JD_REGISTER_TRANSFORM(CustomTestTransform, "CustomTestTransform")
    .SetDeserializer(
        [](const TransformConfig& config,
           absl::Span<const std::shared_ptr<neptune::data::Resource>> resources)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          if (resources.size() != 1) {
            return absl::InvalidArgumentError("Expected exactly one resource");
          }
          auto int_resource =
              std::dynamic_pointer_cast<const IntResource>(resources[0]);
          if (!int_resource) {
            return absl::InvalidArgumentError("Expected IntResource");
          }
          return std::make_unique<CustomTestTransform>(std::move(int_resource));
        });

}  // namespace jax_data::composite
