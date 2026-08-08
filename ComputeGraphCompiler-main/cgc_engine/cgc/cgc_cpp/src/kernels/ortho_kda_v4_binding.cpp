#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <vector>
#include <cstring>
#include "kernels/ortho_kda_v4.cuh"

namespace py = pybind11;

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

class OrthoKDAV4Module {
public:
    OrthoKDAV4Module() : kv_(nullptr), num_heads_(0), head_dim_(0), ortho_base_dim_(128), initialized_(false) {}
    ~OrthoKDAV4Module() { reset(); }

    void init(int num_heads, int head_dim, int ortho_base_dim = 128) {
        reset();
        num_heads_ = num_heads;
        head_dim_ = head_dim;
        ortho_base_dim_ = ortho_base_dim;

#ifdef __CUDACC__
        cudaError_t err = ortho_kda_v4_alloc_kv(&kv_, num_heads_, head_dim_, ortho_base_dim_, 0);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate OrthoKDAKV_v4 on GPU");
        }
#endif
        initialized_ = true;
    }

    void reset() {
#ifdef __CUDACC__
        if (kv_) {
            ortho_kda_v4_reset(kv_, 0);
        }
#endif
        initialized_ = false;
    }

    void free_kv() {
#ifdef __CUDACC__
        if (kv_) {
            ortho_kda_v4_free_kv(kv_, 0);
        }
#endif
        kv_ = nullptr;
        initialized_ = false;
    }

    void update(
        py::array_t<float, py::array::c_style | py::array::forcecast> key,
        py::array_t<float, py::array::c_style | py::array::forcecast> value
    ) {
        if (!initialized_) {
            throw std::runtime_error("OrthoKDA v4 not initialized. Call init() first.");
        }

#ifdef __CUDACC__
        auto key_buf = key.request();
        auto value_buf = value.request();

        const float* key_ptr = static_cast<const float*>(key_buf.ptr);
        const float* value_ptr = static_cast<const float*>(value_buf.ptr);

        ortho_kda_v4_update(kv_, key_ptr, value_ptr, 0);
#endif
    }

    py::array_t<float> forward(
        py::array_t<float, py::array::c_style | py::array::forcecast> Q
    ) {
        if (!initialized_) {
            throw std::runtime_error("OrthoKDA v4 not initialized. Call init() first.");
        }

        auto Q_buf = Q.request();

        auto out = py::array_t<float>(Q_buf.size);
        auto out_buf = out.request();
        float* out_ptr = static_cast<float*>(out_buf.ptr);

        memset(out_ptr, 0, out_buf.size * sizeof(float));

#ifdef __CUDACC__
        const float* Q_ptr = static_cast<const float*>(Q_buf.ptr);
        ortho_kda_v4_forward(kv_, Q_ptr, out_ptr, 0);
#endif

        out.resize(Q_buf.shape);

        return out;
    }

    py::dict get_state() {
        if (!initialized_) {
            throw std::runtime_error("OrthoKDA v4 not initialized.");
        }

        int k_size = ortho_base_dim_ * head_dim_;

        std::vector<float> K(k_size, 0.0f);
        std::vector<float> V(k_size, 0.0f);
        std::vector<float> decay(ortho_base_dim_, 0.0f);
        int idx = 0;

#ifdef __CUDACC__
        if (kv_) {
            ortho_kda_v4_get_state(kv_, K.data(), V.data(), decay.data(), &idx, 0);
        }
#endif

        py::dict state;
        state["K"] = py::array_t<float>({ortho_base_dim_, head_dim_}, K.data());
        state["V"] = py::array_t<float>({ortho_base_dim_, head_dim_}, V.data());
        state["decay"] = py::array_t<float>({ortho_base_dim_}, decay.data());
        state["idx"] = idx;
        state["num_heads"] = num_heads_;
        state["head_dim"] = head_dim_;
        state["ortho_base_dim"] = ortho_base_dim_;

        return state;
    }

    bool is_initialized() const { return initialized_; }
    int num_heads() const { return num_heads_; }
    int head_dim() const { return head_dim_; }
    int ortho_base_dim() const { return ortho_base_dim_; }

private:
    OrthoKDAKV_v4* kv_;
    int num_heads_;
    int head_dim_;
    int ortho_base_dim_;
    bool initialized_;
};

PYBIND11_MODULE(ortho_kda_v4_cpp, m) {
    m.doc() = "Ortho KDA v4 - True Orthogonal Basis Accumulation KDA";

    py::class_<OrthoKDAV4Module>(m, "OrthoKDAV4")
        .def(py::init<>())
        .def("init", &OrthoKDAV4Module::init, "Initialize OrthoKDA v4",
             py::arg("num_heads"), py::arg("head_dim"), py::arg("ortho_base_dim") = 128)
        .def("reset", &OrthoKDAV4Module::reset, "Reset KDA state")
        .def("free", &OrthoKDAV4Module::free_kv, "Free GPU memory")
        .def("update", &OrthoKDAV4Module::update, "Update orthogonal basis",
             py::arg("key"), py::arg("value"))
        .def("forward", &OrthoKDAV4Module::forward, "KDA forward pass",
             py::arg("Q"))
        .def("get_state", &OrthoKDAV4Module::get_state, "Get current state")
        .def_property_readonly("initialized", &OrthoKDAV4Module::is_initialized)
        .def_property_readonly("num_heads", &OrthoKDAV4Module::num_heads)
        .def_property_readonly("head_dim", &OrthoKDAV4Module::head_dim)
        .def_property_readonly("ortho_base_dim", &OrthoKDAV4Module::ortho_base_dim);

    m.def("create_ortho_kda_v4", []() { return new OrthoKDAV4Module(); }, "Create OrthoKDA v4 instance");
    m.def("destroy_ortho_kda_v4", [](OrthoKDAV4Module* kda) { delete kda; }, "Destroy OrthoKDA v4 instance");

    m.def("ortho_kda_v4_benchmark", [](int num_heads, int head_dim, int ortho_base_dim, int iterations) {
        auto kda = OrthoKDAV4Module();
        kda.init(num_heads, head_dim, ortho_base_dim);

        std::vector<float> key(head_dim);
        std::vector<float> value(head_dim);
        std::vector<float> Q(head_dim);

        for (auto& v : key) v = static_cast<float>(rand()) / RAND_MAX * 0.1f;
        for (auto& v : value) v = static_cast<float>(rand()) / RAND_MAX * 0.1f;
        for (auto& v : Q) v = static_cast<float>(rand()) / RAND_MAX * 0.1f;

        py::array_t<float> key_arr(key);
        py::array_t<float> value_arr(value);
        py::array_t<float> Q_arr(Q);

#ifdef __CUDACC__
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int i = 0; i < 10; i++) {
            kda.update(key_arr, value_arr);
        }

        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            kda.forward(Q_arr);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
#else
        float ms = 0.0f;
#endif

        py::dict result;
        result["avg_ms"] = ms / iterations;
        result["iterations"] = iterations;
        result["num_heads"] = num_heads;
        result["head_dim"] = head_dim;
        result["ortho_base_dim"] = ortho_base_dim;

        return result;
    }, "Benchmark OrthoKDA v4", py::arg("num_heads"), py::arg("head_dim"), py::arg("ortho_base_dim") = 128, py::arg("iterations") = 1000);
}
