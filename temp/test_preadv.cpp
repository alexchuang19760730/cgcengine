// [CGC merge-read unit probe] verify preadv(file-contiguous run, scattered dsts) == per-segment
// preads on the real carrier GGUF. Isolates the merge-read corruption seen in steady MTP runs.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <sys/uio.h>
#include <unistd.h>
#include <fcntl.h>

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "/Users/alexchuang/Documents/flashkv0516/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X-headIQ2.gguf";
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    // three contiguous file ranges (arbitrary 90KB-ish segments), scattered dsts
    const off_t base = 0x10000000;
    const size_t seg = 90000;
    uint8_t a[seg], b[seg], c[seg], ref[3 * seg];
    struct iovec iov[3] = { { a, seg }, { b, seg }, { c, seg } };
    ssize_t rd = preadv(fd, iov, 3, base);
    printf("preadv: rd=%zd (want %zu) errno after=%d\n", rd, 3 * seg, errno);
    if (rd != (ssize_t) (3 * seg)) { perror("preadv short"); return 1; }
    ssize_t r2 = pread(fd, ref, 3 * seg, base);
    printf("pread : r2=%zd\n", r2);
    int bad = 0;
    if (memcmp(a, ref + 0 * seg, seg) != 0) bad |= 1;
    if (memcmp(b, ref + 1 * seg, seg) != 0) bad |= 2;
    if (memcmp(c, ref + 2 * seg, seg) != 0) bad |= 4;
    printf("mismatch mask=%d (0 == identical)\n", bad);
    return bad != 0;
}
