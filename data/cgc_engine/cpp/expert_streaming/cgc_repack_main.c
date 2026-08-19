// cgc_repack_main.c — 命令行入口

#include "cgc_repack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    cgc_repack_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.quant_bits = 3; // 默认 IQ3_M
    opts.n_threads = 4;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            opts.input_dir = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            opts.output_dir = argv[++i];
        } else if (strcmp(argv[i], "--imatrix") == 0 && i + 1 < argc) {
            opts.imatrix_path = argv[++i];
        } else if (strcmp(argv[i], "--bits") == 0 && i + 1 < argc) {
            opts.quant_bits = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dry-run") == 0) {
            opts.dry_run = true;
        } else if (strcmp(argv[i], "--overwrite") == 0) {
            opts.overwrite = true;
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            opts.n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: cgc_repack --input <HF模型目录> --output <输出目录> [options]\n");
            printf("Options:\n");
            printf("  --imatrix <file>  use imatrix file for quantization\n");
            printf("  --bits <N>        quantization bits (default: 3 = IQ3_M)\n");
            printf("  --dry-run         only print plan, don't write files\n");
            printf("  --overwrite       overwrite existing output\n");
            printf("  --threads <N>     number of threads (default: 4)\n");
            return 0;
        }
    }

    if (!opts.input_dir || !opts.output_dir) {
        fprintf(stderr, "Error: --input and --output required\n");
        fprintf(stderr, "Run with --help for usage\n");
        return 1;
    }

    return cgc_repack_run(&opts);
}
