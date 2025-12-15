#include <omp.h>

namespace basicnn::linalg {

    namespace {

        struct OMPInitializer {
            
            OMPInitializer() {
                omp_set_dynamic(0);                        // Disable dynamic adjustment
                omp_set_num_threads(omp_get_num_procs());  // Use all logical cores
            }
        };

        // Mark as used so the linker doesn’t optimize it away
        __attribute__((used))
        static OMPInitializer omp_initializer;
    }
}