// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));
        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static void ggml_backend_sched_split_inputs_grow(struct ggml_backend_sched_split * split) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_WARN("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) split->inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

static void ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_WARN("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) sched->graph_inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // [CGC 2026-09-15 S1] The split header is promoted from GGML_LOG_DEBUG to GGML_LOG_WARN on
    // purpose. `sched->debug` is already non-zero only when GGML_SCHED_DEBUG is set, so nothing
    // changes for a normal run, but the default llama log verbosity threshold (INFO) filters
    // GGML_LOG_LEVEL_DEBUG out entirely -- measured: a build with GGML_SCHED_DEBUG=1 printed
    // ZERO "## SPLIT" lines, which is indistinguishable from "there is exactly one split". A
    // diagnostic that can silently print nothing is worse than no diagnostic. The per-node detail
    // below stays at DEBUG (use ---verbose if you need it).
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_WARN("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_WARN(": ");
                }
                GGML_LOG_WARN("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_WARN("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            // [CGC 2026-09-15 S1] promoted from GGML_LOG_DEBUG for the same reason as the split
            // header above: the default llama verbosity threshold hides DEBUG, and a diagnostic
            // that silently prints nothing reads as "nothing to report".
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_WARN("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_WARN(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_WARN("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs >= split->inputs_capacity) {
                        const size_t id = hash_id(src);
                        int src_backend_id = sched->hv_tensor_backend_ids[id];
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            ggml_set_input(tensor_copy);
                            ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        if (n_graph_inputs >= sched->graph_inputs_capacity) {
                            ggml_backend_sched_graph_inputs_grow(sched);
                        }
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    int total_inputs = sched->n_graph_inputs;
    for (int i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // Optimize this split of the graph. This needs to happen before we make graph_copy,
        // so they are in sync.
        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

// [CGC 2026-09-15] `CGC_OA_ASYNC` is a VALUE, not a flag, and until now the gate in
// ggml_backend_sched_compute_splits read it as `getenv("CGC_OA_ASYNC") != nullptr`. So
// CGC_OA_ASYNC=0 -- which run_server.sh emits for every profile that asks for the non-segmented
// path, and which the startup banner then prints as `oa_async=0` -- still selected the SEGMENTED
// branch. An empty string would too, because getenv returns "" and not NULL. The switch had
// therefore been ON in every arm ever run through run_server.sh, including the ones recorded as
// oa_async=0: a "control" that was byte-for-byte the arm under test.
//
// The failure mode is the same one the run_server.sh env allowlist has (an unlisted variable is
// silently dropped, so "no effect" and "not set" are indistinguishable) except inverted: here the
// variable IS set and its VALUE is ignored, which reads as "this dispatcher makes no difference".
// Measured 2026-09-15 21:32: p25-gputime-noasync and p25-slotgpu-noasync reproduced the segmented
// arms' md5 sets exactly ({dc055e63} and {29ca694a, 672585db, b8c705cc}) and the same pool counters
// (misses 13040 / 14296, file_reads 36360 / 39948), i.e. they were re-runs, not controls.
//
// Unset / empty / nonzero first char = segmented. That keeps every recorded digest reproducible
// (prod25 sets "1") while making "0" mean what every caller already documents it to mean.
static bool cgc_oa_async_enabled() {
    const char * e = getenv("CGC_OA_ASYNC");
    if (e == nullptr || e[0] == '\0') {
        return true; // unset: the historical default is the segmented path
    }
    return e[0] != '0';
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                ggml_backend_tensor_copy(input, input_cpy);
            } else {
                // wait for the split backend to finish using the input before overwriting it
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = split->graph.nodes[0];
                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                    )) {

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    ggml_backend_synchronize(input_backend);

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor) {
                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                        ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                        ggml_backend_synchronize(ids_backend);

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                GGML_ASSERT(id >= 0 && id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                            }
                        }

                        prev_ids_tensor = ids_tensor;
                    }

                    // group consecutive experts and copy them together
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        ggml_backend_tensor_set_async(split_backend,
                            input_cpy,
                            (const uint8_t *)input->data + expert_offset, expert_offset,
                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                            // this is necessary for MMQ in the CUDA backend
                            expert_size_copy + padding_end);
                    };

                    int id = 0;
                    while (!ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    int32_t first_id = id;
                    int32_t last_id = first_id;

                    for (++id; id < n_expert; ++id) {
                        if (!ggml_bitset_get(used_ids.data(), id)) {
                            continue;
                        }

                        if (id == last_id + 1) {
                            last_id = id;
                            continue;
                        }

                        copy_experts(first_id, last_id);

                        first_id = id;
                        last_id = id;
                    }
                    copy_experts(first_id, last_id);
                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        ggml_backend_synchronize(input_backend);
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                }
            }
        }

        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        } else if (cgc_oa_async_enabled() &&
                   getenv("CGC_VERIFY_OP_TIMING") == nullptr &&
                   strcmp(ggml_backend_name(split_backend), "CPU") != 0) {
            // CGC: dispatch the Metal split in segments. Segments end at the ARGSORT op (which
            // actually produces the expert ids); the top-k VIEW is a dependency-free alias that
            // ggml may place before its producer, so using it as the boundary would fire the hook
            // before the ids are computed -> garbage routing. Splits without any argsort node
            // (plain graphs) run fully async.
            //
            // DEFAULT (correct): per segment, first WAIT for segment[i] to fully complete (all
            // n_cb+1 cmd buffers, polled non-blocking via ggml_metal_get_cgc_done), then fire the
            // top-k hook which writes the remap leaf, and only THEN submit segment[i+1] (which
            // consumes that remap leaf via mul_mat_id). Submitting segment[i+1] before the hook
            // writes its remap is a CPU/GPU race: the Metal command buffer references the remap
            // buffer and the GPU may read it before the CPU write lands (modifying a buffer of an
            // in-flight command buffer is UB) -> stale remap -> garbage / non-deterministic output.
            // This showed up as short-prompt/cold-cache divergence (slow hook due to cache fills)
            // while warm long prompts happened to win the race and stayed bit-identical.
            // A/B toggle: CGC_SUBMIT_AHEAD=1 restores the racy submit-ahead order (diagnostic).
            // A/B toggle: CGC_TOPK_BOUNDARY=1 reverts to the buggy topk-VIEW boundary (diagnostic).
            const char * cgc_topk_bd = getenv("CGC_TOPK_BOUNDARY");
            const bool use_topk_bd = cgc_topk_bd != nullptr && cgc_topk_bd[0] != 0;
            const char * bd_prefix = use_topk_bd ? "ffn_moe_topk-" : "ffn_moe_argsort-";
            const size_t bd_len = use_topk_bd ? 13 : 16;
            int n_as = 0;
            for (int i = 0; i < split->graph.n_nodes; i++) {
                if (strncmp(split->graph.nodes[i]->name, bd_prefix, bd_len) == 0) {
                    n_as++;
                }
            }
            if (n_as == 0) {
                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }
            } else {
                // segments: [0..as0], [as0+1..as1], ..., [as_{n-1}+1..end]
                // only the first n_as segments carry an argsort node; the tail is plain compute
                const int n_nodes = split->graph.n_nodes;
                int as_idx[64];
                struct ggml_tensor * as_topk[64]; // matching top-k VIEW (same data, ne[0]=8)
                int n_as_found = 0;
                for (int i = 0; i < n_nodes; i++) {
                    const char * nm = split->graph.nodes[i]->name;
                    if (strncmp(nm, bd_prefix, bd_len) == 0) {
                        as_idx[n_as_found] = i;
                        // find the matching top-k view for the same layer
                        struct ggml_tensor * view = nullptr;
                        for (int k = 0; k < n_nodes; k++) {
                            if (strncmp(split->graph.nodes[k]->name, "ffn_moe_topk-", 13) == 0 &&
                                strcmp(split->graph.nodes[k]->name + 13, nm + bd_len) == 0) {
                                view = split->graph.nodes[k];
                                break;
                            }
                        }
                        as_topk[n_as_found] = view;
                        n_as_found++;
                    }
                }
                const int n_segs = n_as_found + 1; // tail segment included

                // poll the Metal backend for completed segments without blocking the pipeline
                auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(split_backend));
                typedef int (*cgc_done_fn)(ggml_backend_t);
                cgc_done_fn cgc_done = (cgc_done_fn) ggml_backend_reg_get_proc_address(reg, "ggml_metal_get_cgc_done");
                typedef int (*cgc_bufs_fn)(ggml_backend_t);
                cgc_bufs_fn cgc_bufs = (cgc_bufs_fn) ggml_backend_reg_get_proc_address(reg, "ggml_metal_get_cgc_bufs");
                const int bufs  = cgc_bufs ? cgc_bufs(split_backend) : 1; // completions per graph_compute (n_cb+1)
                const int done0 = cgc_done ? cgc_done(split_backend) : -1;

                // [CGC GPU-side timing] resolved through the same proc-address mechanism as
                // cgc_done (libggml-metal is a separate dylib; libggml never links it directly).
                // NULL when CGC_GPU_TIMING is unset, in which case the completions never sample.
                typedef int (*cgc_gpu_take_fn)(ggml_backend_t, int64_t *);
                cgc_gpu_take_fn cgc_gpu_take = getenv("CGC_GPU_TIMING") != nullptr
                    ? (cgc_gpu_take_fn) ggml_backend_reg_get_proc_address(reg, "ggml_metal_get_cgc_gpu_take")
                    : nullptr;

                // [CGC 2026-09-18 node-level GPU time] CGC_GPU_NODES=1 attributes each segment's GPU
                // busy time to the NODE KINDS that produced it, one level below the per-layer table.
                //
                // Why this exists: M3_VERDICT closed M3 for lack of an instrument -- `wait` is a
                // CPU-side spin (72 ms/step) and nothing could say whether the GPU was busy inside
                // it or which op was. The per-layer `gpu`/`union` fields (dp_lay_gpu/dp_lay_uni) were
                // the finest GPU-side reading, i.e. 40 buckets. It turns out no Metal sampling is
                // needed to go finer: ggml_metal_graph_compute already splits each segment into
                // n_cb+1 command buffers over FIXED, contiguous node ranges, and Metal records a
                // GPUStartTime/GPUEndTime on each one. So a buffer's duration is attributable to the
                // kinds of node in its range.
                //
                // What this can and cannot separate: it gives a duration per contiguous node RANGE,
                // so it answers "does ffn_moe_* carry a large share of the GPU time" and NOT "node X
                // took Y ms". It does not need an MTLCounterSampleBuffer (whose sample points insert
                // barriers and would perturb the thing being measured), and it adds no state on the
                // Metal side. Both accessors are resolved through the proc-address table because
                // libggml-metal is a separate dylib.
                //
                // Requires CGC_DECODE_PROFILE=1 for the step cadence and for the denominator of the
                // built-in self-check below (the kind table must add up to the same segment total the
                // per-layer `layer gpu_sum` reports; if it does not, the ranges or the buffer/node
                // mapping are wrong and the table means nothing).
                typedef int (*cgc_gpu_take_cb_fn)(ggml_backend_t, int64_t *, int);
                typedef const char * (*cgc_node_name_fn)(ggml_backend_t, int);
                static const bool ns_on = getenv("CGC_GPU_NODES") != nullptr;
                cgc_gpu_take_cb_fn cgc_gpu_take_cb = ns_on
                    ? (cgc_gpu_take_cb_fn) ggml_backend_reg_get_proc_address(reg, "ggml_metal_get_cgc_gpu_take_cb")
                    : nullptr;
                cgc_node_name_fn cgc_node_name = ns_on
                    ? (cgc_node_name_fn) ggml_backend_reg_get_proc_address(reg, "ggml_metal_cgc_node_name")
                    : nullptr;
                // [CGC 2026-09-18 OP-KEYED attribution] CGC_GPU_OPS=1 adds a second table keyed by
                // the ggml OP instead of by name prefix, because the name-keyed one cannot answer
                // the question that decides whether "fuse the shape chain" is worth doing: DOES A
                // VIEW COST GPU TIME? A name bucket mixes ops (`ffn_moe_` holds GLU/MUL/SUM_ROWS/
                // DIV/VIEW/RESHAPE), and node COUNT is not GPU TIME. With the op known, buffers
                // whose nodes are ALL the same op give an EXACT per-node cost (duration / nodes)
                // with no model and no fit -- the positive control the least-squares attempt lacked.
                // Requires CGC_GPU_NODES (that is what fills the node snapshot).
                typedef int (*cgc_node_op_fn)(ggml_backend_t, int);
                static const bool ns_ops = ns_on && getenv("CGC_GPU_OPS") != nullptr;
                // [CGC 2026-09-18 WORK-WEIGHTED NAME TABLE] Resolved under CGC_GPU_NODES, NOT under
                // CGC_GPU_OPS. The name table's split weight now needs the op of every node it
                // buckets, and gating that on a second env var would make the work-weighted column
                // silently ABSENT in exactly the arm that reads the name table (`en-nodes`) -- the
                // same "unset == not set" confusion that already cost one round with
                // CGC_VERIFY_OP_TIMING and one with CGC_GRPH_DBG. `ns_ops` still gates the OP-KEYED
                // table itself, which is a different question ("which op costs what").
                cgc_node_op_fn cgc_node_op = ns_on
                    ? (cgc_node_op_fn) ggml_backend_reg_get_proc_address(reg, "ggml_metal_cgc_node_op")
                    : nullptr;
                // [CGC 2026-09-18] Does this op put anything into the command buffer? PROVEN, not
                // modelled: the Metal encoder switches on node->op and treats NONE / RESHAPE / VIEW /
                // TRANSPOSE / PERMUTE as "noop -> next node" (ggml-metal-ops.cpp:242-252), so a buffer
                // holding only those ops contains ZERO GPU commands. Two things follow, and both are
                // measured facts rather than opinions: (a) a node-COUNT split is wrong in a way that
                // matters, because ~39% of a layer's nodes are these five; (b) the per-command-buffer
                // durations can NOT be attributed to nodes at all -- an all-VIEW buffer was reported
                // at 592 us/node, which is impossible for a buffer with no commands in it.
                // So this predicate is what the split weight uses instead of a count.
                auto cgc_op_emits_work = [](int op) -> bool {
                    return op != GGML_OP_NONE && op != GGML_OP_RESHAPE && op != GGML_OP_VIEW &&
                           op != GGML_OP_TRANSPOSE && op != GGML_OP_PERMUTE;
                };
                // The kind table. 48 named kinds + 1 implicit "(other)"; the name is the node name
                // with its trailing "-<layer>" removed, so it keeps full node-kind resolution.
                static char    ns_kind_nm[48][48] = {{0}};
                static int64_t ns_kind_ns[48]     = {0};
                static int     ns_kind_n          = 0;
                static int64_t ns_total           = 0;
                static int64_t ns_other           = 0;
                static int     ns_print_n         = 0;
                // [CGC 2026-09-18 node-level GPU time TRACE] CGC_GPU_NODES_TRACE=1 prints the RAW
                // identity of the hottest command buffers plus a range-size histogram.
                //
                // Why the kind table alone is not enough: it cannot separate "the MoE matmuls are
                // not named ffn_moe_*" from "they share a command buffer with several small nodes,
                // so the node-count split hands them only 1/N of their own buffer's duration".
                // Measured motivation: the first table had `(other)` 16.1% / `node` 13.6% /
                // `cache` 12.8% and no ffn_moe_gate row -- and it printed only the top 14 of 22
                // kinds, so 26.9% of the time was in rows that were never shown. Printing the
                // range's real names AND its size separates the two hypotheses in one line.
                static const bool ns_trace = getenv("CGC_GPU_NODES_TRACE") != nullptr;
                static int64_t ns_hot_dur[3] = {0, 0, 0};
                static int     ns_hot_nn[3]  = {0, 0, 0};   // names captured
                static int     ns_hot_rs[3]  = {0, 0, 0};   // nodes in the range
                static char    ns_hot_nm[3][8][48] = {{{0}}};
                static int64_t ns_rng1 = 0, ns_rng2 = 0, ns_rng3_7 = 0, ns_rng8p = 0;
                // [CGC 2026-09-18 node-level GPU time BOUNDS] The node-count split inside one
                // command buffer is a guess, and the TRACE run showed how bad a guess it can be:
                // ranges reach 64 nodes (measured histogram 1=9 2=26 3-7=263 8+=40), so a buffer
                // holding one ffn_moe_gate GEMV plus 63 small nodes handed the GEMV 1/64 of its own
                // duration. Rather than invent a better split, bracket it -- M3's decision rule is a
                // threshold test, so bounds are enough:
                //   UB = sum of the durations of every buffer that CONTAINS the kind   (upper bound)
                //   LB = sum of the durations of buffers where the kind is the ONLY kind
                //        (a solo buffer's duration is exactly that kind's duration -- a lower bound)
                // A kind whose LB is already above the threshold has passed the test; one whose UB
                // is below it has failed. Only when the threshold falls inside [LB, UB] does the
                // question need finer machinery.
                static int64_t ns_kind_ub[48] = {0};
                static int64_t ns_kind_lb[48] = {0};
                // [CGC 2026-09-19 KIND x OP] Per-(kind, op) work-weighted duration. The kind
                // table's biggest row is `node` -- ggml's auto-name for UNNAMED nodes
                // (ggml.c:7192) -- and a NAME is not an identity, the OP is. Reset with the other
                // ns_kind_* accumulators, after the print that consumes it.
                static int64_t ns_kop_wns[48][GGML_OP_COUNT] = {{0}};
                // [CGC 2026-09-18 WORK-WEIGHTED NAME TABLE] The same partition as ns_kind_ns but with
                // the buffer's duration shared over the NAMED nodes whose op ENCODES something
                // (cgc_op_emits_work) instead of over every named node. This is what turns the name
                // table -- `ffn_moe_*` vs `cache` vs `attn_*` -- from a direction into a ranking,
                // and it fixes the defect by construction rather than by tuning: a node-count split
                // divides a kind by ~64 whenever the kind shares the segment's main-thread buffer
                // (n_main = MAX(64, 0.1*n_nodes), ggml-metal-context.m:1099), and ~39% of a layer's
                // nodes are no-ops that pad that denominator without contributing any GPU command.
                // MEASURED (op-keyed table, same build): the count-weighted share of MUL_MAT fell
                // 27.3% -> 11.1% when the slices narrowed, while its work-weighted share stayed
                // 8.0% -> 8.4%. So wcntw is the column that survives a change of granularity.
                static int64_t ns_kind_wns[48] = {0};
                // Sum of the durations of the buffers that contained >=1 named work node -- i.e. the
                // mass the work-weighted column is a partition OF. ns_total - ns_kind_work_ns is the
                // part of "segment busy" that NO named kind can claim, and printing it is what keeps
                // the column honest instead of hiding redistribution in plain sight.
                static int64_t ns_kind_work_ns = 0;
                static int64_t ns_kind_work_nd = 0;

                // [CGC 2026-09-18 OP-KEYED attribution] see CGC_GPU_OPS above. The decisive column is
                // `uni` = (sum of the durations of buffers whose nodes are ALL this op) / (the nodes
                // in those buffers). That is an EXACT per-node cost for the op -- no model, no fit --
                // and it is the only thing in this instrument that can say whether the ~40% of a
                // layer that is VIEW/RESHAPE/PERMUTE/PAD costs anything at all. `nd` counts nodes
                // (ALL of them, including the empty-named ones the name table skips, so `nodes_named`
                // and `nodes_all` are printed separately as a cross-check on the two loops).
                static char    nsop_nm[64][24] = {{0}};
                static int     nsop_op[64]     = {0};   // the ggml_op for each slot (for the weight)
                static int64_t nsop_ns[64]     = {0};   // count-weighted share (the name table's cntw)
                static int64_t nsop_wns[64]    = {0};   // WORK-weighted share (see cgc_op_emits_work)
                static int64_t nsop_ub[64]     = {0};   // buffers containing the op
                static int64_t nsop_lb[64]     = {0};   // buffers whose nodes are ALL this op
                static int64_t nsop_uni_ns[64] = {0};   // duration of those same all-one-op buffers
                static int64_t nsop_uni_nd[64] = {0};   // ... and how many nodes they held
                static int64_t nsop_nd[64]     = {0};   // nodes of this op (denominator of `uni`)
                static int     nsop_n          = 0;
                static int64_t nsop_total      = 0;     // sum of durations over buffers with >=1 node
                static int64_t nsop_nodes      = 0;
                static int64_t nsop_work_nd    = 0;     // ... of which this many actually encode
                static int64_t ns_kind_nodes   = 0;     // same thing for the name table (cross-check)

                // [CGC M0 decode profile 2026-09-13] Per-layer attribution of a decode step. The
                // segmented loop below serializes GPU layer i -> CPU top-k hook -> submit of layer
                // i+1, so the step wall is sum(wait + cb + submit) over layers. The CGC-SEG print
                // averages over 160 segments (~4 steps), which cannot answer "which layer
                // dominates" -- and the churn data says the answer is NOT uniform (layers 1-2 carry
                // the highest demand distinct-expert counts). Inert unless CGC_DECODE_PROFILE.
                // NOTE: only reachable under CGC_OA_ASYNC=1; without it the whole graph is one async
                // submit and none of these three components is separable.
                static const bool dp_on  = getenv("CGC_DECODE_PROFILE") != nullptr;
                static const int  dp_all = []() { const char * e = getenv("CGC_DECODE_PROFILE_ALL"); return e != nullptr ? atoi(e) : 0; }();
                static int64_t dp_last_submit_us = 0;  // pending submit, consumed by the next hook
                static int64_t dp_lay_w[64]   = {0};   // GPU wait, per layer
                static int64_t dp_lay_cb[64]  = {0};   // top-k hook (slot mgmt + blocking fill)
                static int64_t dp_lay_sub[64] = {0};   // submit of that layer's segment
                static int64_t dp_lay_n[64]   = {0};   // segments observed, per layer
                // [CGC M3 2026-09-17] The GPU clock, attributed per layer. `cgc_gpu_take`
                // already hands back per-segment Metal timestamps, but until now they were only
                // summed into the per-step `gt_*` accumulators and the layer identity was dropped,
                // so the only GPU-side reading in this repo was per step. `dp_lay_gpu` is the sum
                // of GPUEndTime-GPUStartTime over the segment's buffers, `dp_lay_uni` its span, and
                // `dp_lay_gap` the idle window between the previous segment's end and this one's
                // start (the same quantity CGC-GPUTIME reports per step as `gap`). All three are
                // Metal-side readings: they add no work to the graph, which is what makes this the
                // one attribution here that is perturbation-free. `dp_lay_sg` counts segments that
                // actually carried a usable timestamp, so "zero" can be told from "not measured".
                static int64_t dp_lay_gpu[64] = {0};   // GPU busy (sum over the segment's buffers)
                static int64_t dp_lay_uni[64] = {0};   // GPU union (span)
                static int64_t dp_lay_gap[64] = {0};   // GPU idle before this layer's segment
                static int64_t dp_lay_sg[64]  = {0};   // segments with a usable timestamp
                static int64_t dp_step        = 0;     // graph_computes since start
                static int64_t dp_ntok        = 0;     // tokens in the graph being profiled (its own shape)

                // [CGC 2026-09-15 GPU-side timing] Accumulators for CGC_GPU_TIMING (see
                // ggml-metal-context.m for why). gpu_* come from the Metal command buffers'
                // own GPUStartTime/GPUEndTime, read at each segment boundary right after the
                // cgc_done poll succeeded. `wait` is the same CPU-side window CGC-DECPROF
                // reports, so gpu_busy_sum/wait is directly the answer to "is the 91% wait real
                // GPU execution or launch + completion latency?".
                //   busy_sum: Σ(end-start) over the segment's n_cb+1 buffers (overlap counted twice)
                //   union:    max(end)-min(start) -- << busy_sum means the buffers DO overlap
                //   gap:      Σ(start_i - end_{i-1}) in the GPU clock => GPU idle between segments.
                //             gt_prev_end is cleared per graph, so gap never spans a
                //             draft->verify or step->step transition.
                static int64_t gt_busy = 0, gt_union = 0, gt_gap = 0, gt_wait = 0;
                static int64_t gt_nseg = 0, gt_nbuf = 0, gt_nstep = 0, gt_unsup = 0, gt_prev_end = -1;

                auto seg_view = [&](int s) {
                    const int a = (s == 0) ? 0 : (as_idx[s-1] + 1);
                    const int b = (s == n_segs-1) ? n_nodes - 1 : as_idx[s];
                    struct ggml_cgraph gv = ggml_graph_view(&split->graph, a, b + 1);
                    return gv;
                };

                // submit the first segment, then pipeline: submit seg[i+1] ahead, wait for seg[i],
                // fire the top-k hook (writes the remap leaf) that seg[i+1] consumes
                const int64_t dp_v0 = dp_on ? ggml_time_us() : 0;
                struct ggml_cgraph gv0 = seg_view(0);
                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv0);
                if (dp_on) { dp_last_submit_us = ggml_time_us() - dp_v0; }
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }
                static int cgc_grph_dbg_n = 0;
                if (getenv("CGC_GRPH_DBG") != nullptr && cgc_grph_dbg_n < 6) {
                    cgc_grph_dbg_n++;
                    fprintf(stderr, "CGC-GRPH-BEGIN n_nodes=%d backend=%s\n", split->graph.n_nodes, ggml_backend_name(split_backend));
                    for (int gi = 0; gi < split->graph.n_nodes; gi++) {
                        ggml_tensor * gn = split->graph.nodes[gi];
                        fprintf(stderr, "CGC-GRPH[%d] name=%s op=%d(%s) ne=[%lld,%lld]\n",
                                gi, gn->name, (int) gn->op, ggml_op_name(gn->op),
                                (long long) gn->ne[0], (long long) gn->ne[1]);
                    }
                }
                static int64_t p_us = 0, p_n = 0;
                const bool submit_dbg = getenv("CGC_SUBMIT_DBG") != nullptr;
                // CGC raciness fix: submit segment[i+1] only AFTER the top-k hook of segment[i]
                // has written its remap leaf (which segment[i+1] consumes via mul_mat_id), so the
                // remap buffer is stable before the command buffer referencing it is committed.
                // CGC_SUBMIT_AHEAD=1 restores the old racy submit-ahead order for A/B perf compare.
                const bool submit_ahead = getenv("CGC_SUBMIT_AHEAD") != nullptr;
                auto submit_seg = [&](int s) -> enum ggml_status {
                    const int64_t v0 = ggml_time_us();
                    const int d0 = submit_dbg && cgc_done ? cgc_done(split_backend) : -1;
                    struct ggml_cgraph gv = seg_view(s);
                    enum ggml_status ec2 = ggml_backend_graph_compute_async(split_backend, &gv);
                    if (ec2 != GGML_STATUS_SUCCESS) {
                        return ec2;
                    }
                    const int64_t v1 = ggml_time_us();
                    dp_last_submit_us = v1 - v0;   // [CGC M0] consumed by the next top-k hook
                    const int d1 = submit_dbg && cgc_done ? cgc_done(split_backend) : -1;
                    if (submit_dbg && (s % 40) == 0) {
                        fprintf(stderr, "CGC-SUBMIT: seg=%d dur=%lld us gpu_adv=%d\n",
                                s, (long long) (v1 - v0), d1 - d0);
                    }
                    p_us += v1 - v0;
                    p_n++;
                    return GGML_STATUS_SUCCESS;
                };
                auto hook_seg = [&](int i) -> bool {
                    static int64_t w_us = 0, c_us = 0, n = 0;
                    const int64_t st0 = ggml_time_us();
                    if (cgc_done) {
                        // wait for segment i's WHOLE run: all n_cb+1 cmd buffers of segments 0..i
                        // (one completion per buffer). Waiting only on the main buffer (done0+i+1)
                        // fired the top-k hook while the argsort was still running -> stale ids ->
                        // garbage remap -> whole-graph corruption (echo prompt / all-'!').
                        const int target = done0 + (i + 1) * bufs;
                        while (cgc_done(split_backend) < target) {
                            sched_yield();
                        }
                    } else {
                        ggml_backend_synchronize(split_backend);
                    }
                    const int64_t st1 = ggml_time_us();
                    // [CGC GPU-side timing] the poll above just observed segment i's last
                    // completion, so all of segment i's command buffers are completed -- the
                    // only point where Metal reports GPUStartTime/GPUEndTime. Segment i+1 has
                    // not been submitted yet, so nothing else can be in flight.
                    int64_t sg_busy = 0, sg_union = 0, sg_gap = 0;
                    if (cgc_gpu_take != nullptr) {
                        int64_t g[5] = {0, 0, 0, 0, 0};
                        const int gns = cgc_gpu_take(split_backend, g);
                        sg_busy  = g[0];
                        sg_union = g[1];
                        gt_busy  += g[0];
                        gt_union += g[1];
                        gt_unsup += g[4];
                        gt_nbuf  += gns;
                        if (gns > 0) {
                            gt_nseg++;
                        }
                        // GPU-clock idle between the previous segment's end and this one's start.
                        // This window sits inside the previous segment's hook+submit (CPU) time,
                        // so gap vs (cb+submit) is a built-in cross-check on both instruments.
                        if (g[2] > 0 && gt_prev_end > 0 && g[2] > gt_prev_end) {
                            gt_gap += g[2] - gt_prev_end;
                            sg_gap  = g[2] - gt_prev_end;
                        }
                        if (g[3] > 0) {
                            gt_prev_end = g[3];
                        }
                        gt_wait += st1 - st0;
                    }
                    // [CGC 2026-09-18 node-level GPU time] Attribute THIS segment's GPU busy time to
                    // the node kinds that produced it -- read at the same instant as cgc_gpu_take
                    // above, where segment i's command buffers are still the ones sitting in the
                    // Metal context's cmd_bufs[] and segment i+1 has not been submitted yet. Each
                    // buffer covers a contiguous node range, so its duration is split across the
                    // kinds in that range in proportion to the node counts. Summing the buffers is a
                    // BUSY measure (they overlap), which is exactly what the per-layer `gpu` field
                    // reports -- hence the self-check in the printer.
                    if (cgc_gpu_take_cb != nullptr && cgc_node_name != nullptr) {
                        // Deep enough for the per-node mode (CGC_CB_N_MAIN=1 + CGC_N_CB=127 gives
                        // n_cb+1 = 128 buffers); a smaller buffer silently DROPS the tail slots,
                        // and the tail is where DPROF's self-check denominator comes from.
                        int64_t cb_rec[5 * 129];
                        const int n_rec = cgc_gpu_take_cb(split_backend, cb_rec, 129);
                        for (int r = 0; r < n_rec; r++) {
                            const int64_t * rec = cb_rec + 5*r;
                            if (rec[4] == 0) {
                                continue;   // slot with no completed buffer / no usable timestamp
                            }
                            const int64_t dur = rec[1] - rec[0];
                            const int nd_a = (int) rec[2];
                            const int nd_b = (int) rec[3];
                            int cnt[49] = {0};
                            int tot = 0;
                            int wcnt[49] = {0};   // same buckets, restricted to nodes whose op encodes
                            int wtot = 0;
                            // [CGC 2026-09-19 KIND x OP] The work-weighted split is per node, so the
                            // pair list is per buffer; bounded by the buffer's working nodes.
                            int kop_ix[160];
                            int kop_op[160];
                            int kop_k = 0;
                            for (int nd = nd_a; nd < nd_b; nd++) {
                                const char * nm = cgc_node_name(split_backend, nd);
                                if (nm == nullptr || nm[0] == '\0') {
                                    continue;
                                }
                                // Bucket the node into a FIXED kind vocabulary. "Strip the trailing
                                // -<layer>" alone leaves ~600 distinct names on a 40-layer model, so
                                // a first-come table overflows long before the big kinds arrive --
                                // MEASURED: the first revision of this instrument put 67% of the
                                // segment busy time in "(other)" because its 48 slots had been taken
                                // by rare warmup-only kinds (conv_states_reshaped, alpha, beta...).
                                // The vocabulary answers M3's question -- the ffn_moe_* share against
                                // the attention share -- rather than trying to be exhaustive.
                                static const char * const ns_fix[] = {
                                    // [CGC 2026-09-18 NAMING] three buckets that only exist because
                                    // the builders were given names for previously-unnamed tensors:
                                    // ffn_moe_add = the (n_expert_used-1) intermediate terms of the
                                    // MoE expert-output reduce (llama-graph.cpp:2790/2799), and
                                    // gdn_state / gdn_out_raw = ggml_gated_delta_net
                                    // (models/delta-net-base.cpp:402/572). Before naming they were
                                    // 240 + 30 nodes inside `node`, which was the largest single
                                    // unnamed cluster in the whole table.
                                    "ffn_moe_add", "gdn_state", "gdn_out",
                                    "ffn_moe_argsort", "ffn_moe_logits", "ffn_moe_probs", "ffn_moe_slots",
                                    "ffn_moe_topk", "ffn_moe_gate_up", "ffn_moe_gate", "ffn_moe_up",
                                    "ffn_moe_down", "ffn_moe_", "ffn_gate", "ffn_up", "ffn_down", "ffn_",
                                    // [CGC 2026-09-18] `top_k` is a REAL MoE op (ggml TOP_K, ne=[256,2]
                                    // over the expert logits) that had no entry, so its duration was
                                    // being reported as "(other)" -- while `ffn_moe_topk` IS in the
                                    // vocabulary but names a VIEW (op=38, ne=[8,2], proven from the
                                    // CGC-GRPH dump: ffn_moe_topk-0..N are all VIEW, the computation is
                                    // ffn_moe_argsort-* ARGSORT). So the bucket that looks like the
                                    // MoE selection step reported wcntw 0.00 BY CONSTRUCTION -- it is a
                                    // view -- and the cost that actually exists sat in "(other)".
                                    // Bucket name != operation; this is the second time that bit.
                                    "top_k",
                                    "linear_attn", "attn_q", "attn_k", "attn_v", "attn_output",
                                    "attn_norm", "attn_post_norm", "attn_residual", "attn_inp_k_rot",
                                    "attn_inp_v_rot", "attn_inp_kq_mask", "attn_", "rope", "soft_max",
                                    "rms_norm", "norm", "get_rows", "mul_mat", "cpy", "concat", "add",
                                    "leaf", "node", "cache", "conv", "result",
                                    // [CGC 2026-09-18 NAMING round 2] `(other)` was still 853 nodes -- and
                                    // it is NOT unnamed tensors. It is NAMED tensors whose prefixes were
                                    // missing from this list. Measured from a FRESH CGC-GRPH dump
                                    // (4116 nodes, log llama_server_20260918_050331; the 02:53 dump is
                                    // stale -- it predates the ffn_moe_add / gdn_* naming and still
                                    // shows 620 in `node` where the current tree shows 350):
                                    //   shared_expert_gate(+_sigmoid) 80 | delta-net gates/state
                                    //   (alpha, beta, beta_sigmoid, a_softplus, state_predelta, z-) 190 |
                                    //   delta-net conv preprocessing (q_conv, k_conv, v_conv) 150 |
                                    //   attention QKV (Qcur*, Kcur*, Vcur, kqv_out) 100 |
                                    //   qkv_mixed 30 | gate 40 | l_out 40 | final_output 30 | h_nextn 1.
                                    // Only the ~62 nodes with an EMPTY base name stay in `(other)`: they are
                                    // views of tensors nobody named, so they need a builder fix
                                    // (ggml_set_name), not a vocabulary entry. That is the `node` bucket's
                                    // job -- the two halves have DIFFERENT fixes and must not be confused.
                                    //
                                    // Prefix choice notes: matching is longest-prefix-wins, so `ffn_gate`
                                    // and `shared_expert_gate` cannot collide with `gate`; and `z-` (not
                                    // `z`) keeps a one-character prefix from swallowing future names.
                                    "shared_expert_gate", "qkv_mixed", "state_predelta", "a_softplus",
                                    "q_conv", "k_conv", "v_conv", "alpha", "beta", "z-",
                                    "Qcur", "Kcur", "Vcur", "kqv_out", "final_output", "l_out", "gate",
                                    "h_nextn",
                                    // [CGC 2026-09-18 §EN-149] Names added to the graph so the
                                    // CGC-GPUNODE `node` bucket can be split: before this, both landed in
                                    // `node` (or `(other)` once named but unlisted) and could not be ranked.
                                    "dnqkv_proj", "dnbeta_proj", "dn_normg_mul",
                                };
                                const int ns_nfix = (int) (sizeof(ns_fix) / sizeof(ns_fix[0]));
                                const char * key = NULL;
                                size_t key_len = 0;
                                for (int q = 0; q < ns_nfix; q++) {
                                    const size_t lq = strlen(ns_fix[q]);
                                    if (lq > key_len && strncmp(nm, ns_fix[q], lq) == 0) {
                                        key = ns_fix[q];
                                        key_len = lq;
                                    }
                                }
                                if (key == NULL) {
                                    key = "(other)";
                                }
                                int ix = -1;
                                for (int q = 0; q < ns_kind_n; q++) {
                                    if (strcmp(ns_kind_nm[q], key) == 0) { ix = q; break; }
                                }
                                if (ix < 0 && ns_kind_n < 48) {
                                    ix = ns_kind_n++;
                                    snprintf(ns_kind_nm[ix], sizeof(ns_kind_nm[ix]), "%s", key);
                                }
                                cnt[ix < 0 ? 48 : ix]++;
                                tot++;
                                // WORK-WEIGHTED companion count, taken in the SAME pass so that both
                                // denominators describe the same node set (named nodes only -- the
                                // name table's partition must still sum to the buffer's duration).
                                if (cgc_node_op != nullptr) {
                                    const int nop = cgc_node_op(split_backend, nd);
                                    if (nop >= 0 && cgc_op_emits_work(nop)) {
                                        const int wi = ix < 0 ? 48 : ix;
                                        wcnt[wi]++;
                                        wtot++;
                                        if (wi < 48 && nop < GGML_OP_COUNT && kop_k < 160) {
                                            kop_ix[kop_k] = wi;
                                            kop_op[kop_k] = nop;
                                            kop_k++;
                                        }
                                    }
                                }
                            }
                            ns_total += dur;
                            if (tot == 0) {
                                continue;
                            }
                            for (int q = 0; q < 49; q++) {
                                if (cnt[q] == 0) {
                                    continue;
                                }
                                const int64_t share = dur * cnt[q] / tot;
                                if (q < 48) { ns_kind_ns[q] += share; } else { ns_other += share; }
                            }
                            // [CGC 2026-09-18 BOUNDS] Per-kind lower/upper bounds (see the statics),
                            // plus -- under TRACE -- the range-size histogram and the hottest
                            // ranges' RAW node names. The names are copied here because the name
                            // snapshot belongs to the most recent graph_compute, which by print time
                            // is a LATER segment than the one being described.
                            for (int q = 0; q < 49; q++) {
                                if (cnt[q] == 0 || q >= 48) {
                                    continue;
                                }
                                ns_kind_ub[q] += dur;
                                if (cnt[q] == tot) {
                                    ns_kind_lb[q] += dur;   // solo buffer: its duration IS this kind's
                                }
                            }
                            ns_kind_nodes += tot;
                            // [CGC 2026-09-18 WORK-WEIGHTED SPLIT] Same buffer, same set of named
                            // nodes, but only the ones that encode anything are in the denominator.
                            // A buffer with NO such node contributes nothing here -- its duration
                            // stays in ns_total and surfaces as the printed residual -- and that is
                            // correct, not a gap: by the encoder's own switch
                            // (ggml-metal-ops.cpp:242-252) such a buffer contains zero GPU commands,
                            // so its timestamp cannot be anyone's work (MEASURED: an all-VIEW buffer
                            // reported 592 us/node while a 2048-wide ADD reported 6.7).
                            if (wtot > 0) {
                                ns_kind_work_ns += dur;
                                ns_kind_work_nd += wtot;
                                for (int q = 0; q < 49; q++) {
                                    if (wcnt[q] == 0 || q >= 48) {
                                        continue;
                                    }
                                    ns_kind_wns[q] += dur * wcnt[q] / wtot;
                                }
                                // [CGC 2026-09-19 KIND x OP] The identical share, one term per
                                // working node, tagged by (kind, op). Summing a kind's ops
                                // reproduces its wcntw row -- this refines that column, it does
                                // not add a second model.
                                for (int k = 0; k < kop_k; k++) {
                                    ns_kop_wns[kop_ix[k]][kop_op[k]] += dur / wtot;
                                }
                            }
                            // [CGC 2026-09-18 OP-KEYED attribution] independent second pass over the
                            // SAME range, bucketed by the ggml op. Deliberately a separate loop: the
                            // name loop above SKIPS empty-named nodes (a name is what it needs), while
                            // the op of such a node is still perfectly well defined -- so the two
                            // tables legitimately have different denominators, and printing both
                            // counts is what keeps that honest instead of hidden.
                            if (ns_ops && cgc_node_op != nullptr) {
                                int ocnt[64] = {0};
                                int otot = 0;
                                int owork = 0;   // nodes whose op actually emits GPU work
                                for (int nd = nd_a; nd < nd_b; nd++) {
                                    const int op = cgc_node_op(split_backend, nd);
                                    if (op < 0) {
                                        continue;
                                    }
                                    const char * onm = ggml_op_name((enum ggml_op) op);
                                    if (onm == nullptr) {
                                        continue;
                                    }
                                    int oix = -1;
                                    for (int q = 0; q < nsop_n; q++) {
                                        if (strcmp(nsop_nm[q], onm) == 0) { oix = q; break; }
                                    }
                                    if (oix < 0 && nsop_n < 64) {
                                        oix = nsop_n++;
                                        snprintf(nsop_nm[oix], sizeof(nsop_nm[oix]), "%s", onm);
                                        nsop_op[oix] = op;
                                    }
                                    if (oix < 0) {
                                        continue;
                                    }
                                    ocnt[oix]++;
                                    otot++;
                                    if (cgc_op_emits_work(op)) {
                                        owork++;
                                    }
                                }
                                if (otot > 0) {
                                    nsop_total += dur;
                                    nsop_nodes += otot;
                                    nsop_work_nd += owork;
                                    int odistinct = 0;
                                    for (int q = 0; q < nsop_n; q++) {
                                        if (ocnt[q] > 0) { odistinct++; }
                                    }
                                    for (int q = 0; q < nsop_n; q++) {
                                        if (ocnt[q] == 0) {
                                            continue;
                                        }
                                        nsop_nd[q] += ocnt[q];
                                        nsop_ns[q] += dur * ocnt[q] / otot;
                                        // WORK-WEIGHTED: the buffer's duration shared out over only
                                        // the nodes whose op encodes something. Justified by the
                                        // encoder, not by a model -- see cgc_op_emits_work.
                                        if (owork > 0 && cgc_op_emits_work(nsop_op[q])) {
                                            nsop_wns[q] += dur * ocnt[q] / owork;
                                        }
                                        nsop_ub[q] += dur;
                                        if (odistinct == 1) {
                                            // ALL nodes of this buffer are the same op: its whole
                                            // duration belongs to that op, so the per-node cost is an
                                            // exact division and not a share. A buffer of pure no-op
                                            // nodes therefore reports a duration that CANNOT be its
                                            // own work -- which is what turned out to be the case
                                            // (VIEW 592 us/node) and why `uni` is not usable.
                                            nsop_lb[q]     += dur;
                                            nsop_uni_ns[q] += dur;
                                            nsop_uni_nd[q] += ocnt[q];
                                        }
                                    }
                                }
                            }
                            // [CGC 2026-09-18 MATRIX] CGC_GPU_NODES_MATRIX=1 dumps one line per command
                            // buffer -- its duration plus how many nodes of each kind it encoded -- so
                            // the per-kind cost can be recovered OFFLINE by least squares instead of
                            // GUESSED by node count. Why a guess is not good enough: n_main =
                            // MAX(64, 0.1*n_nodes) (ggml-metal-context.m:1099) means EVERY segment's
                            // main-thread buffer holds >= 64 nodes, so any kind living in it is
                            // divided by >= 64 by a count-weighted split -- MEASURED: ffn_moe_gate
                            // cntw 1.91 ms against ub 122.37 ms out of 271.94 ms of segment busy
                            // time. The model fitted offline is dur_i = sum_k cnt_ik * t_k (t_k = one
                            // node of kind k), which is well posed because the small worker buffers
                            // (263 of 360 had 3-7 nodes) isolate kinds the big ones cannot.
                            static const bool ns_matrix = getenv("CGC_GPU_NODES_MATRIX") != nullptr;
                            if (ns_matrix) {
                                fprintf(stderr, "CGC-NSM a=%d b=%d dur_ns=%lld nk=%d",
                                        nd_a, nd_b, (long long) dur, tot);
                                for (int q = 0; q < ns_kind_n; q++) {
                                    if (cnt[q] > 0) {
                                        fprintf(stderr, " %s:%d", ns_kind_nm[q], cnt[q]);
                                    }
                                }
                                if (cnt[48] > 0) {
                                    fprintf(stderr, " (other):%d", cnt[48]);
                                }
                                fprintf(stderr, "\n");
                            }
                            if (ns_trace) {
                                const int rsz = nd_b - nd_a;
                                if (rsz <= 1)      { ns_rng1++; }
                                else if (rsz == 2) { ns_rng2++; }
                                else if (rsz < 8)  { ns_rng3_7++; }
                                else               { ns_rng8p++; }
                                // Keep the three slots SORTED. The first revision shifted whenever a
                                // candidate beat the LAST slot, so the slots held the right set only
                                // coincidentally and the display order was insertion order --
                                // measured: hot#2 0.050 ms printed above hot#3 3.386 ms.
                                if (ns_hot_dur[2] == 0 || dur > ns_hot_dur[2]) {
                                    int pos = 2;
                                    while (pos > 0 &&
                                           (ns_hot_dur[pos-1] == 0 || dur > ns_hot_dur[pos-1])) {
                                        ns_hot_dur[pos] = ns_hot_dur[pos-1];
                                        ns_hot_nn[pos]  = ns_hot_nn[pos-1];
                                        ns_hot_rs[pos]  = ns_hot_rs[pos-1];
                                        memcpy(ns_hot_nm[pos], ns_hot_nm[pos-1], sizeof(ns_hot_nm[pos]));
                                        pos--;
                                    }
                                    ns_hot_dur[pos] = dur;
                                    ns_hot_rs[pos]  = rsz;
                                    ns_hot_nn[pos]  = rsz < 8 ? rsz : 8;
                                    for (int q = 0; q < ns_hot_nn[pos]; q++) {
                                        const char * nz = cgc_node_name(split_backend, nd_a + q);
                                        snprintf(ns_hot_nm[pos][q], sizeof(ns_hot_nm[pos][q]), "%s",
                                                 nz != nullptr ? nz : "(null)");
                                    }
                                }
                            }
                        }
                    }
                    // [CGC bit-bisect v7] in-compute tensor dump: forward every node of the
                    // just-completed segment to the eval callback (ask=false). Segments 0..i
                    // have completed and segment i+1 has not been submitted yet, so every
                    // node's output buffer still holds its computed value — unlike a post-
                    // compute dump, where later segments have already recycled those buffers.
                    // The llama_context side (expert_cache_eval_cb -> cgc_tdcb_maybe_dump)
                    // filters by exact tensor name (CGC_TD_CB) and writes the data. Nodes named
                    // ffn_moe_topk* are skipped here: the dedicated call right below fires those.
                    if (getenv("CGC_TD_CB") != nullptr) {
                        const int a0 = (i == 0) ? 0 : (as_idx[i-1] + 1);
                        for (int k = a0; k <= as_idx[i]; k++) {
                            struct ggml_tensor * tn = split->graph.nodes[k];
                            if (tn == nullptr || tn->name[0] == '\0') {
                                continue;
                            }
                            if (strncmp(tn->name, "ffn_moe_topk", 12) == 0) {
                                continue;
                            }
                            sched->callback_eval(tn, false, sched->callback_eval_user_data);
                        }
                    }
                    struct ggml_tensor * ttopk = as_topk[i];
                    if (ttopk == nullptr) {
                        ttopk = split->graph.nodes[as_idx[i]];
                    }
                    if (!sched->callback_eval(ttopk, false, sched->callback_eval_user_data)) {
                        return false;
                    }
                    const int64_t st2 = ggml_time_us();
                    // [CGC M0 decode profile] attribute this layer's wait / hook / submit. The
                    // submit consumed here is the one that queued THIS layer's segment.
                    if (dp_on) {
                        // [CGC prefill-visible profile] The token count of THIS graph, read off the
                        // top-k tensor (ne = [n_expert_used, n_tokens]). It is recorded because
                        // `dp_step == 1` is only the *assumption* that the first graph is the
                        // prefill; a warmup graph would silently take that slot and the profile
                        // would then be describing decode while claiming prefill. Printing the
                        // shape makes each line state its own provenance: ntok=2048 => prefill,
                        // ntok=1 => decode. See eng-src-0011.
                        const int64_t dp_t = ttopk != nullptr ? ttopk->ne[1] : 0;
                        if (dp_t > dp_ntok) {
                            dp_ntok = dp_t;
                        }
                        const char * dp_dash = ttopk != nullptr ? strrchr(ttopk->name, '-') : nullptr;
                        const int dp_il = dp_dash != nullptr ? atoi(dp_dash + 1) : i;
                        if (dp_il >= 0 && dp_il < 64) {
                            dp_lay_w[dp_il]   += st1 - st0;
                            dp_lay_cb[dp_il]  += st2 - st1;
                            dp_lay_sub[dp_il] += dp_last_submit_us;
                            dp_lay_gpu[dp_il] += sg_busy;
                            dp_lay_uni[dp_il] += sg_union;
                            dp_lay_gap[dp_il] += sg_gap;
                            if (sg_busy > 0) {
                                dp_lay_sg[dp_il]++;
                            }
                            dp_lay_n[dp_il]++;
                        }
                        dp_last_submit_us = 0;
                    }
                    w_us += st1 - st0;
                    c_us += st2 - st1;
                    n++;
                    if (n % 160 == 0) {
                        fprintf(stderr, "CGC-SEG: wait %.1f cb %.1f submit %.1f us (%d)\n",
                                (double) w_us / n, (double) c_us / n, (double) p_us / p_n, (int) n);
                    }
                    return true;
                };
                for (int i = 0; i < n_segs; i++) {
                    if (submit_ahead && i + 1 < n_segs) {
                        ec = submit_seg(i + 1);
                        if (ec != GGML_STATUS_SUCCESS) {
                            return ec;
                        }
                    }
                    if (i < n_as_found) {
                        if (!hook_seg(i)) {
                            break;
                        }
                    }
                    if (!submit_ahead && i + 1 < n_segs) {
                        ec = submit_seg(i + 1);
                        if (ec != GGML_STATUS_SUCCESS) {
                            return ec;
                        }
                    }
                }

                // [CGC M0 decode profile] One line per 8 steps with the wait/cb/submit split, the
                // top-8 layers by (wait + cb), and -- with CGC_DECODE_PROFILE_ALL=1 -- every layer.
                // Reset each step: these are per-step attributions, not run totals.
                if (dp_on) {
                    dp_step++;
                    int64_t dp_w = 0, dp_cb = 0, dp_sb = 0;
                    int dp_layers = 0;
                    for (int l = 0; l < 64; l++) {
                        if (dp_lay_n[l] == 0) {
                            continue;
                        }
                        dp_layers++;
                        dp_w  += dp_lay_w[l];
                        dp_cb += dp_lay_cb[l];
                        dp_sb += dp_lay_sub[l];
                    }
                    const int64_t dp_tot = dp_w + dp_cb + dp_sb;
                    // Emit on the decode cadence, on the first graph of the process, and on ANY
                    // batched graph. The last two conditions are what make this instrument able to
                    // answer a prefill question at all: with `n_prompt=2048` and `-ub 6144` the whole
                    // prefill is a single graph_compute (dp_step=1), so the original
                    // `(dp_step % 8) == 0` never fired for it and the accumulators were zeroed
                    // before any later step could see them -- prefill was structurally invisible.
                    // See eng-src-0011 (and the 95-log "min step is always 8" fingerprint).
                    if (dp_tot > 0 && ((dp_step % 8) == 0 || dp_step == 1 || dp_ntok > 1)) {
                        const double dp_inv = 100.0 / (double) dp_tot;
                        // The gt_* family is in NANOSECONDS (CGC-GPUTIME divides by 1e6), while
                        // the dp_lay_* family above is in MICROSECONDS. Mixing the two units is how
                        // the first revision of this line printed a 1760x too large number; the
                        // cross-check that catches it is `gpu_sum` vs CGC-GPUTIME's gpu_busy_sum.
                        int64_t dp_gs = 0, dp_gu = 0, dp_gg = 0;
                        for (int li = 0; li < 64; li++) {
                            dp_gs += dp_lay_gpu[li];
                            dp_gu += dp_lay_uni[li];
                            dp_gg += dp_lay_gap[li];
                        }
                        char dp_gpu_tail[160];
                        dp_gpu_tail[0] = '\0';
                        if (cgc_gpu_take != nullptr) {
                            snprintf(dp_gpu_tail, sizeof(dp_gpu_tail),
                                     " | layer gpu_sum=%.2f union_sum=%.2f gap_sum=%.2f ms%s",
                                     (double) dp_gs / 1e6, (double) dp_gu / 1e6,
                                     (double) dp_gg / 1e6,
                                     dp_gs == 0 ? " (NO TIMESTAMPS)" : "");
                        }
                        fprintf(stderr,
                                "CGC-DECPROF: step=%lld segs=%d layers=%d total=%.2f ms | "
                                "wait=%.2f (%.0f%%) cb=%.2f (%.0f%%) submit=%.2f (%.0f%%) ntok=%lld%s\n",
                                (long long) dp_step, n_segs, dp_layers, (double) dp_tot / 1000.0,
                                (double) dp_w / 1000.0, (double) dp_w * dp_inv,
                                (double) dp_cb / 1000.0, (double) dp_cb * dp_inv,
                                (double) dp_sb / 1000.0, (double) dp_sb * dp_inv,
                                (long long) dp_ntok, dp_gpu_tail);
                        bool dp_used[64] = {false};
                        for (int rank = 0; rank < 8; rank++) {
                            int dp_best = -1;
                            int64_t dp_bestv = 0;
                            for (int l = 0; l < 64; l++) {
                                if (dp_used[l] || dp_lay_n[l] == 0) {
                                    continue;
                                }
                                const int64_t v = dp_lay_w[l] + dp_lay_cb[l];
                                if (dp_best < 0 || v > dp_bestv) {
                                    dp_best  = l;
                                    dp_bestv = v;
                                }
                            }
                            if (dp_best < 0) {
                                break;
                            }
                            dp_used[dp_best] = true;
                            fprintf(stderr, "CGC-DECPROF top%d: L%d wait=%.2f cb=%.2f submit=%.2f ms "
                                    "gpu=%.2f union=%.2f gap=%.2f sg=%lld n=%lld\n",
                                    rank + 1, dp_best,
                                    (double) dp_lay_w[dp_best] / 1000.0,
                                    (double) dp_lay_cb[dp_best] / 1000.0,
                                    (double) dp_lay_sub[dp_best] / 1000.0,
                                    (double) dp_lay_gpu[dp_best] / 1e6,
                                    (double) dp_lay_uni[dp_best] / 1e6,
                                    (double) dp_lay_gap[dp_best] / 1e6,
                                    (long long) dp_lay_sg[dp_best],
                                    (long long) dp_lay_n[dp_best]);
                        }
                        if (dp_all != 0) {
                            for (int l = 0; l < 64; l++) {
                                if (dp_lay_n[l] == 0) {
                                    continue;
                                }
                                fprintf(stderr, "CGC-DECPROF all: L%d wait=%.2f cb=%.2f submit=%.2f ms "
                                        "gpu=%.2f union=%.2f gap=%.2f sg=%lld n=%lld\n",
                                        l,
                                        (double) dp_lay_w[l] / 1000.0,
                                        (double) dp_lay_cb[l] / 1000.0,
                                        (double) dp_lay_sub[l] / 1000.0,
                                        (double) dp_lay_gpu[l] / 1e6,
                                        (double) dp_lay_uni[l] / 1e6,
                                        (double) dp_lay_gap[l] / 1e6,
                                        (long long) dp_lay_sg[l],
                                        (long long) dp_lay_n[l]);
                            }
                        }
                        // [CGC 2026-09-18 node-level GPU time] the per-KIND table for this step.
                        // SELF-CHECK: `seg_busy` must equal the `layer gpu_sum` printed on the
                        // CGC-DECPROF line above, because both sum the same per-segment Metal busy
                        // time. A non-zero `delta` means the node ranges or the buffer/node mapping
                        // are wrong and the table below means nothing -- the same "verify the
                        // instrument before the number" rule that CGC-GPUTIME's `unsupported=` field
                        // exists for. Printed on a 1-in-8 step sample: with MTP on the DECPROF
                        // cadence fires every step and 48 kind lines per step would swamp the log.
                        if (ns_on && ns_total > 0 && (ns_print_n++ % 8) == 0) {
                            fprintf(stderr, "CGC-GPUNODE: step=%lld seg_busy=%.2f ms | layer gpu_sum=%.2f ms "
                                    "| delta=%.2f%% | other=%.2f ms | nkind=%d\n",
                                    (long long) dp_step, (double) ns_total / 1e6, (double) dp_gs / 1e6,
                                    dp_gs > 0 ? 100.0 * (double) (ns_total - dp_gs) / (double) dp_gs : 0.0,
                                    (double) ns_other / 1e6, ns_kind_n);
                            // ALL kinds, not a top-N: the first revision printed 14 of 22 and the
                            // hidden 8 carried 26.9% of the step -- so a kind that was present but
                            // ranked below the cut was indistinguishable from a kind that was
                            // missing, which is exactly the question this table exists to answer.
                            // Rank by UB, not by the count-weighted value: a kind that the split
                            // diluted has a LARGE ub and a small cntw, so ranking by cntw is exactly
                            // how ffn_moe_gate stayed out of sight. Columns: cntw = the count-weighted
                            // guess, lb = solo-buffer sum (lower bound), ub = sum over buffers that
                            // contain the kind (upper bound). The truth is in [lb, ub].
                            for (int rank = 0; rank < 48; rank++) {
                                int best = -1;
                                int64_t best_v = 0;
                                for (int q = 0; q < ns_kind_n; q++) {
                                    if (ns_kind_ub[q] > best_v) { best = q; best_v = ns_kind_ub[q]; }
                                }
                                if (best < 0) {
                                    break;
                                }
                                // 100*v/total. The first revision wrote v*(100/total)/1e6 -- one
                                // 1e6 too many, because ns_total is in NANOSECONDS while v had
                                // already been divided by 1e6 -- and printed 0.0% for every row.
                                const double pc = 100.0 / (double) ns_total;
                                // `wcntw` first because it is the column to rank by: the count-
                                // weighted `cntw` is the guess that n_main's >=64-node floor made
                                // useless, `lb`/`ub` bracket the truth without needing a weight at
                                // all, and wcntw splits each buffer over the nodes that actually
                                // encode. The header's work-attributed line prints the mass this is a
                                // partition of, so the residual is visible instead of redistributed.
                                fprintf(stderr, "CGC-GPUNODE:   %-24s wcntw=%7.2f %5.1f%% | cntw=%7.2f %5.1f%% | lb=%7.2f %5.1f%% | ub=%7.2f %5.1f%%\n",
                                        ns_kind_nm[best],
                                        (double) ns_kind_wns[best] / 1e6, (double) ns_kind_wns[best] * pc,
                                        (double) ns_kind_ns[best] / 1e6, (double) ns_kind_ns[best] * pc,
                                        (double) ns_kind_lb[best] / 1e6, (double) ns_kind_lb[best] * pc,
                                        (double) best_v / 1e6,           (double) best_v * pc);
                                ns_kind_ub[best] = 0;   // consumed by this print
                            }
                            // The work-attributed mass and the residual. Without this line the wcntw
                            // column would read as a partition of the whole step, when it is only a
                            // partition of the buffers that contained a named work node. The residual
                            // is the same quantity the op-keyed table reports as
                            // `total - sum(wcntw)` (10.6% of segment busy at n_cb=16, 51.7% at
                            // n_cb=63), and it GROWS with the number of command buffers -- which is
                            // why the residual is the thing that says "do not quote an absolute
                            // seg_busy", not the column itself.
                            fprintf(stderr,
                                    "CGC-GPUNODE: work-attributed %.2f of %.2f ms (%.1f%%) | "
                                    "named_work_nodes=%lld | residual=%.2f ms (%.1f%%)\n",
                                    (double) ns_kind_work_ns / 1e6, (double) ns_total / 1e6,
                                    ns_total > 0 ? 100.0 * (double) ns_kind_work_ns / (double) ns_total : 0.0,
                                    (long long) ns_kind_work_nd,
                                    (double) (ns_total - ns_kind_work_ns) / 1e6,
                                    ns_total > 0 ? 100.0 * (double) (ns_total - ns_kind_work_ns) / (double) ns_total : 0.0);
                            // [CGC 2026-09-19 KIND x OP] A kind NAME is a label; the OP is the work.
                            // This is the table that turns one into the other, and it is the only way
                            // to answer "the top kind is `node`, so which ops is it?" -- `node_<i>`
                            // is what the TRACE prints, and that is not an answer. Rows below 0.5% of
                            // their kind are suppressed; a kind's rows sum to its wcntw entry.
                            if (ns_ops) {
                                for (int q = 0; q < 48; q++) {
                                    int64_t kop_tot = 0;
                                    for (int o = 0; o < GGML_OP_COUNT; o++) { kop_tot += ns_kop_wns[q][o]; }
                                    if (kop_tot == 0) { continue; }
                                    for (int o = 0; o < GGML_OP_COUNT; o++) {
                                        if (ns_kop_wns[q][o] * 200 < kop_tot) { continue; }
                                        const char * kop_onm = ggml_op_name((enum ggml_op) o);
                                        fprintf(stderr, "CGC-GPUOPK: %-22s %-14s %8.2f ms %5.1f%% of kind | %5.1f%% of step\n",
                                                ns_kind_nm[q], kop_onm == nullptr ? "?" : kop_onm,
                                                (double) ns_kop_wns[q][o] / 1e6,
                                                100.0 * (double) ns_kop_wns[q][o] / (double) kop_tot,
                                                ns_total > 0 ? 100.0 * (double) ns_kop_wns[q][o] / (double) ns_total : 0.0);
                                    }
                                }
                            }
                            // ... and the ranking the name table exists for. The same rows, ordered by
                            // the column that survives a change of granularity, so `ffn_moe_* vs cache
                            // vs attn_*` can be READ OFF directly instead of inferred from a list
                            // ordered by an upper bound (where a kind present in many buffers
                            // outranks a kind that is actually expensive).
                            for (int rank = 0; rank < 10; rank++) {
                                int best = -1;
                                int64_t best_v = 0;
                                for (int q = 0; q < ns_kind_n; q++) {
                                    if (ns_kind_wns[q] > best_v) { best = q; best_v = ns_kind_wns[q]; }
                                }
                                if (best < 0) {
                                    break;
                                }
                                // Only the work-weighted column here: ns_kind_ub has already been
                                // consumed (zeroed) by the upper-bound-ordered loop above, so a `ub=`
                                // field in this line would print 0.00 for every row -- the shape of
                                // the same mistake this instrument already paid for once (a consumed
                                // array read twice).
                                fprintf(stderr, "CGC-GPUNODE:  *bywork %-20s wcntw=%7.2f %5.1f%%\n",
                                        ns_kind_nm[best], (double) best_v / 1e6,
                                        ns_total > 0 ? 100.0 * (double) best_v / (double) ns_total : 0.0);
                                ns_kind_wns[best] = 0;   // consumed by this print
                            }
                            // [CGC 2026-09-18 OP-KEYED table] one row per ggml op. Read it as: `nd` =
                            // how many nodes of this op, `uni` = the EXACT us/node from buffers that
                            // held only this op (blank when no such buffer was seen -- then the row
                            // carries only bounds), `cntw` = the count-weighted share, `lb`/`ub` = the
                            // bounds. The number that answers "does the shape chain cost anything" is
                            // `uni` for VIEW/RESHAPE/PERMUTE against `uni` for MUL_MAT_ID.
                            if (ns_ops) {
                                fprintf(stderr,
                                        "CGC-GPUOPS: step=%lld total=%.2f ms nodes_all=%lld "
                                        "nodes_work=%lld nodes_named=%lld nop=%d\n",
                                        (long long) dp_step, (double) nsop_total / 1e6,
                                        (long long) nsop_nodes, (long long) nsop_work_nd,
                                        (long long) ns_kind_nodes, nsop_n);
                                for (int rank = 0; rank < 24; rank++) {
                                    int best = -1;
                                    int64_t best_v = 0;
                                    for (int q = 0; q < nsop_n; q++) {
                                        if (nsop_ub[q] > best_v) { best = q; best_v = nsop_ub[q]; }
                                    }
                                    if (best < 0) {
                                        break;
                                    }
                                    const double pc  = nsop_total > 0 ? 100.0 / (double) nsop_total : 0.0;
                                    const double uni = nsop_uni_nd[best] > 0
                                        ? (double) nsop_uni_ns[best] / (double) nsop_uni_nd[best] / 1e3
                                        : -1.0;   // us per node; -1 = never seen in an all-one-op buffer
                                    // `wcntw` = the same share but weighted by "this op encodes
                                    // something" instead of by node count. For a no-op op it is 0 by
                                    // construction, which is the point: a VIEW cannot cost GPU time.
                                    fprintf(stderr,
                                            "CGC-GPUOPS:   %-16s nd=%6lld %s wcntw=%8.2f %5.1f%%  "
                                            "cntw=%8.2f %5.1f%%  ub=%8.2f %5.1f%%  uni=%8.3f\n",
                                            nsop_nm[best], (long long) nsop_nd[best],
                                            cgc_op_emits_work(nsop_op[best]) ? "work" : "NOOP",
                                            (double) nsop_wns[best] / 1e6, (double) nsop_wns[best] * pc,
                                            (double) nsop_ns[best] / 1e6, (double) nsop_ns[best] * pc,
                                            (double) best_v / 1e6, (double) best_v * pc, uni);
                                    nsop_ub[best] = 0;   // consumed by this print
                                }
                            }
                            if (ns_trace) {
                                fprintf(stderr, "CGC-GPUNODE: range sizes 1=%lld 2=%lld 3-7=%lld 8+=%lld\n",
                                        (long long) ns_rng1, (long long) ns_rng2,
                                        (long long) ns_rng3_7, (long long) ns_rng8p);
                                for (int q = 0; q < 3; q++) {
                                    if (ns_hot_dur[q] == 0) {
                                        continue;
                                    }
                                    fprintf(stderr, "CGC-GPUNODE: hot#%d dur=%.3f ms range=%d nodes:",
                                            q + 1, (double) ns_hot_dur[q] / 1e6, ns_hot_rs[q]);
                                    for (int z = 0; z < ns_hot_nn[q]; z++) {
                                        fprintf(stderr, " %s", ns_hot_nm[q][z]);
                                    }
                                    fprintf(stderr, "\n");
                                }
                            }
                        }
                    }
                    for (int l = 0; l < 64; l++) {
                        dp_lay_w[l] = dp_lay_cb[l] = dp_lay_sub[l] = dp_lay_n[l] = 0;
                        dp_lay_gpu[l] = dp_lay_uni[l] = dp_lay_gap[l] = dp_lay_sg[l] = 0;
                    }
                    ns_total = 0;
                    ns_other = 0;
                    ns_hot_dur[0] = ns_hot_dur[1] = ns_hot_dur[2] = 0;
                    ns_hot_nn[0]  = ns_hot_nn[1]  = ns_hot_nn[2]  = 0;
                    ns_hot_rs[0]  = ns_hot_rs[1]  = ns_hot_rs[2]  = 0;
                    ns_rng1 = ns_rng2 = ns_rng3_7 = ns_rng8p = 0;
                    for (int q = 0; q < ns_kind_n; q++) {
                        ns_kind_ns[q] = 0;
                        ns_kind_ub[q] = 0;
                        ns_kind_lb[q] = 0;
                        ns_kind_wns[q] = 0;
                        // [CGC 2026-09-19 KIND x OP] per-STEP, like every other accumulator here.
                        // It must NOT live in the print block: the print fires every 8th step, so
                        // that version accumulated 8 steps against a 1-step ns_total and reported
                        // `node` at 177x its kind-table value (measured, first run).
                        for (int o = 0; o < GGML_OP_COUNT; o++) { ns_kop_wns[q][o] = 0; }
                    }
                    ns_kind_work_ns = 0;
                    ns_kind_work_nd = 0;
                    // the op table keeps its name slots (nsop_nm/nsop_n) across steps -- like
                    // ns_kind_nm -- so the accumulator indices stay stable run to run.
                    for (int q = 0; q < nsop_n; q++) {
                        nsop_ns[q]     = 0;
                        nsop_wns[q]    = 0;
                        nsop_ub[q]     = 0;
                        nsop_lb[q]     = 0;
                        nsop_uni_ns[q] = 0;
                        nsop_uni_nd[q] = 0;
                        nsop_nd[q]     = 0;
                    }
                    nsop_total    = 0;
                    nsop_nodes    = 0;
                    nsop_work_nd  = 0;
                    ns_kind_nodes = 0;
                    dp_ntok = 0;   // per-step attribution: the next graph states its own shape
                }

                // [CGC GPU-side timing] One line per 4 qualifying graph_computes. The decision
                // number is gpu_busy_sum / wait: >=70% => the wait is real GPU execution (batch
                // the per-expert GEMVs); <=40% => it is launch/completion latency (remove the
                // GPU->CPU->GPU round trip at the segment boundary). gpu_union << gpu_busy_sum
                // additionally says the n_cb+1 buffers of a segment genuinely run concurrently,
                // which is the assumption behind every n_cb tuning result.
                // `skipped` counts buffers with no usable timestamp: it must stay at 0 or the
                // platform is not reporting them and the whole line is meaningless.
                // Only graphs with a real MoE layer count qualify: the MTP draft context runs
                // through here too with a ~2-segment graph, and averaging two different machines
                // together would produce a number that describes neither.
                if (cgc_gpu_take != nullptr) {
                    if (n_as_found >= 5 && gt_nseg > 0) {
                        gt_nstep++;
                        if ((gt_nstep % 4) == 0) {
                            const double w  = (double) gt_wait  / 1e3;
                            const double b  = (double) gt_busy  / 1e6;
                            const double u  = (double) gt_union / 1e6;
                            const double gp = (double) gt_gap   / 1e6;
                            const double pc = w > 0.0 ? 100.0 / w : 0.0;
                            fprintf(stderr,
                                    "CGC-GPUTIME: step=%lld segs=%lld bufs=%lld skipped=%lld "
                                    "wait=%.2f gpu_busy_sum=%.2f (%.0f%%) gpu_union=%.2f (%.0f%%) "
                                    "gap=%.2f (%.0f%%) ms\n",
                                    (long long) gt_nstep, (long long) gt_nseg, (long long) gt_nbuf,
                                    (long long) gt_unsup, w,
                                    b, b * pc, u, u * pc, gp, gp * pc);
                        }
                    }
                    gt_busy = gt_union = gt_gap = gt_wait = 0;
                    gt_nseg = gt_nbuf = gt_unsup = 0;
                    gt_prev_end = -1;
                }

                // CGC: measure how much tail (post top-k) GPU work remains un-waited after the loop
                if (getenv("CGC_TAIL_DBG") != nullptr) {
                    const int64_t tl0 = ggml_time_us();
                    ggml_backend_synchronize(split_backend);
                    fprintf(stderr, "CGC-TAIL: tail_sync=%dus\n", (int)(ggml_time_us() - tl0));
                }
                // [CGC bit-bisect v7] tail segment (everything after the last argsort): the
                // loop above never hooks it. Wait for it to complete, then forward its nodes
                // the same way so post-MoE tensors (ffn_out / l_out / output head) are dumped
                // with fresh values too. Debug-only (CGC_TD_CB): serializes the async pipeline.
                if (getenv("CGC_TD_CB") != nullptr && n_as_found > 0) {
                    if (cgc_done) {
                        const int target = done0 + n_segs * bufs;
                        while (cgc_done(split_backend) < target) {
                            sched_yield();
                        }
                    } else {
                        ggml_backend_synchronize(split_backend);
                    }
                    for (int k = as_idx[n_as_found - 1] + 1; k < n_nodes; k++) {
                        struct ggml_tensor * tn = split->graph.nodes[k];
                        if (tn == nullptr || tn->name[0] == '\0') {
                            continue;
                        }
                        if (strncmp(tn->name, "ffn_moe_topk", 12) == 0) {
                            continue;
                        }
                        sched->callback_eval(tn, false, sched->callback_eval_user_data);
                    }
                }
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            const int64_t cgc_t0 = ggml_time_us();
            const bool cgc_verify_op_timing = getenv("CGC_VERIFY_OP_TIMING") != nullptr;
            auto cgc_verify_op_kind = [](const struct ggml_tensor * node) -> const char * {
                if (node == nullptr || node->name[0] == '\0') {
                    return nullptr;
                }
                if (strncmp(node->name, "ffn_moe_gate-", 13) == 0) {
                    return "gate";
                }
                if (strncmp(node->name, "ffn_moe_up-", 11) == 0) {
                    return "up";
                }
                if (strncmp(node->name, "shared_expert_gate_sigmoid-", sizeof("shared_expert_gate_sigmoid-") - 1) == 0 ||
                    strncmp(node->name, "mtp_shared_expert_gate_sigmoid-", sizeof("mtp_shared_expert_gate_sigmoid-") - 1) == 0) {
                    return "shared_gate_sigmoid";
                }
                if (strncmp(node->name, "ffn_moe_down-", 13) == 0) {
                    return "down";
                }
                return nullptr;
            };
            auto cgc_verify_op_ntok = [](const struct ggml_tensor * node) -> int64_t {
                if (node == nullptr) {
                    return 0;
                }
                if (node->src[2] != nullptr && node->src[2]->type == GGML_TYPE_I32 && node->src[2]->ne[1] > 0) {
                    return node->src[2]->ne[1];
                }
                if (node->ne[2] > 0) {
                    return node->ne[2];
                }
                if (node->ne[1] > 0) {
                    return node->ne[1];
                }
                return 0;
            };
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                const int64_t cgc_chunk_t0 = ggml_time_us();
                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);
                const int64_t cgc_chunk_us = ggml_time_us() - cgc_chunk_t0;

                if (cgc_verify_op_timing && need) {
                    const char * kind = cgc_verify_op_kind(t);
                    const int64_t ntok = cgc_verify_op_ntok(t);
                    if (kind != nullptr && ntok > 1) {
                        static int cgc_verify_op_n = 0;
                        if (cgc_verify_op_n < 4096) {
                            cgc_verify_op_n++;
                            struct ggml_tensor * first = split->graph.nodes[j0];
                            fprintf(stderr,
                                    "CGC-VERIFY-OP: kind=%s us=%lld ntok=%lld span=%d j0=%d j1=%d first=%s last=%s backend=%s\n",
                                    kind,
                                    (long long) cgc_chunk_us,
                                    (long long) ntok,
                                    j1 - j0 + 1,
                                    j0,
                                    j1,
                                    first && first->name[0] ? first->name : "-",
                                    t->name[0] ? t->name : "-",
                                    ggml_backend_name(split_backend));
                        }
                    }
                }

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
            static int64_t cgc_cpu_us = 0;
            static int64_t cgc_cpu_n  = 0;
            cgc_cpu_us += ggml_time_us() - cgc_t0;
            cgc_cpu_n++;
            if (cgc_cpu_n % 160 == 0) {
                fprintf(stderr, "CGC-CPU-SPLIT: avg %.1f us/split (%d)\n",
                        (double) cgc_cpu_us / cgc_cpu_n, (int) cgc_cpu_n);
            }
        }

        // record the event of this copy
        if (split->n_inputs > 0) {
            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (struct ggml_tensor **) calloc(sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    for (int i = 0; i < sched->splits_capacity; i++) {
        free(sched->splits[i].inputs);
    }
    free(sched->splits);
    free(sched->graph_inputs);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
