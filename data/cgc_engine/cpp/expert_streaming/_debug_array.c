#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static uint64_t read_u64(FILE* f) {
    uint8_t b[8];
    if (fread(b, 1, 8, f) != 8) return 0;
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (i * 8);
    return v;
}

static uint32_t read_u32(FILE* f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) v |= (uint32_t)b[i] << (i * 8);
    return v;
}

static char* read_string(FILE* f) {
    uint64_t len = read_u64(f);
    if (len > 65536) {
        fseek(f, (long)len, SEEK_CUR);
        return NULL;
    }
    char* s = (char*)malloc((size_t)len + 1);
    if (!s) return NULL;
    if (len > 0) {
        if (fread(s, 1, (size_t)len, f) != (size_t)len) {
            free(s);
            return NULL;
        }
    }
    s[len] = '\0';
    return s;
}

int main() {
    const char* path = "C:\\Users\\alexchuang\\Desktop\\fastprefill\\gemma4_gguf\\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf";
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open file\n"); return 1; }
    
    fseek(f, 828, SEEK_SET);
    
    uint32_t elem_type = read_u32(f);
    uint64_t n = read_u64(f);
    printf("ARRAY: elem_type=%u, n=%llu, pos=%ld\n", elem_type, (unsigned long long)n, ftell(f));
    
    for (uint64_t i = 0; i < n; i++) {
        long pos_before = ftell(f);
        char* s = read_string(f);
        long pos_after_read = ftell(f);
        
        printf("  [%llu] pos_before=%ld, pos_after_read=%ld", (unsigned long long)i, pos_before, pos_after_read);
        
        if (s) {
            printf(", str='%s'", s);
            free(s);
        } else {
            printf(", read_string returned NULL");
        }
        
        // 4-byte alignment
        long align = (4 - (pos_after_read % 4)) % 4;
        if (align > 0) {
            fseek(f, align, SEEK_CUR);
            printf(", aligned by %d to pos=%ld", (int)align, ftell(f));
        }
        printf("\n");
    }
    
    printf("Final pos: %ld\n", ftell(f));
    
    fclose(f);
    return 0;
}