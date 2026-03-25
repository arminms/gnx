/* The MIT License

   Copyright (c) 2008 Broad Institute / Massachusetts Institute of Technology
                 2011, 2012 Attractive Chaos <attractor@live.co.uk>

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

/* The BGZF library was originally written by Bob Handsaker from the Broad
 * Institute. It was later improved by the SAMtools developers.
 *
 * This is a header-only version: all function definitions are marked static
 * so that each translation unit gets its own private copy.  This eliminates
 * the need for a separately compiled bgzf.c object or any extra CMake target.
 */

#ifndef __BGZF_H
#define __BGZF_H

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility>
#include <sys/types.h>
#include <zlib.h>

// unistd.h provides ssize_t / off_t on POSIX; on Windows use a compat shim
#if !defined(_WIN32)
#include <unistd.h>
#else
#include <io.h>
typedef SSIZE_T ssize_t;
#endif

#ifdef BGZF_MT
#include <pthread.h>
#endif

#define BGZF_BLOCK_SIZE 0xff00 // make sure compressBound(BGZF_BLOCK_SIZE) < BGZF_MAX_BLOCK_SIZE
#define BGZF_MAX_BLOCK_SIZE 0x10000

#define BGZF_ERR_ZLIB 1
#define BGZF_ERR_HEADER 2
#define BGZF_ERR_IO 4
#define BGZF_ERR_MISUSE 8

typedef struct
{
    int errcode : 16, is_write : 2, is_be : 2, compress_level : 12;
    int cache_size;
    int block_length, block_offset;
    int64_t block_address;
    void *uncompressed_block, *compressed_block;
    void *cache; // a pointer to a hash table
    void *fp;    // actual file handler; FILE* on writing; FILE* or knetFile* on reading
#ifdef BGZF_MT
    void *mt; /* only used for multi-threading */
#endif
} BGZF;

#ifndef KSTRING_T
#define KSTRING_T kstring_t
typedef struct __kstring_t
{
    size_t l, m;
    char *s;
} kstring_t;
#endif

#ifndef kroundup32
#define kroundup32(x) (--(x), (x) |= (x) >> 1, (x) |= (x) >> 2, (x) |= (x) >> 4, (x) |= (x) >> 8, (x) |= (x) >> 16, ++(x))
#endif

/* =========================================================================
 * Declarations (all static -- internal linkage per TU)
 * ========================================================================= */

#ifdef __cplusplus
extern "C"
{
#endif

    /******************
     * Basic routines *
     ******************/

    /**
     * Open an existing file descriptor for reading or writing.
     *
     * @param fd    file descriptor
     * @param mode  mode matching /[rwu0-9]+/: 'r' for reading, 'w' for writing and a digit specifies
     *              the zlib compression level; if both 'r' and 'w' are present, 'w' is ignored.
     * @return      BGZF file handler; 0 on error
     */
    static BGZF *bgzf_dopen(int fd, const char *mode);

#define bgzf_fdopen(fd, mode) bgzf_dopen((fd), (mode)) /* for backward compatibility */

    /**
     * Open the specified file for reading or writing.
     */
    static BGZF *bgzf_open(const char *path, const char *mode);

    /**
     * Close the BGZF and free all associated resources.
     *
     * @param fp    BGZF file handler
     * @return      0 on success and -1 on error
     */
    static int bgzf_close(BGZF *fp);

    /**
     * Read up to _length_ bytes from the file storing into _data_.
     *
     * @param fp     BGZF file handler
     * @param data   data array to read into
     * @param length size of data to read
     * @return       number of bytes actually read; 0 on end-of-file and -1 on error
     */
    static ssize_t bgzf_read(BGZF *fp, void *data, size_t length);

    /**
     * Write _length_ bytes from _data_ to the file.
     *
     * @param fp     BGZF file handler
     * @param data   data array to write
     * @param length size of data to write
     * @return       number of bytes actually written; -1 on error
     */
    static ssize_t bgzf_write(BGZF *fp, const void *data, size_t length);

    /**
     * Write the data in the buffer to the file.
     */
    static int bgzf_flush(BGZF *fp);

    /**
     * Return a virtual file pointer to the current location in the file.
     * No interpetation of the value should be made, other than a subsequent
     * call to bgzf_seek can be used to position the file at the same point.
     * Return value is non-negative on success.
     */
    #define bgzf_tell(fp) ((((BGZF *)fp)->block_address << 16) | (((BGZF *)fp)->block_offset & 0xFFFF))

    /**
     * Set the file to read from the location specified by _pos_.
     *
     * @param fp     BGZF file handler
     * @param pos    virtual file offset returned by bgzf_tell()
     * @param whence must be SEEK_SET
     * @return       0 on success and -1 on error
     */
    static int64_t bgzf_seek(BGZF *fp, int64_t pos, int whence);

    /**
     * Check if the BGZF end-of-file (EOF) marker is present
     *
     * @param fp    BGZF file handler opened for reading
     * @return      1 if EOF is present; 0 if not or on I/O error
     */
    static int bgzf_check_EOF(BGZF *fp);

    /**
     * Check if a file is in the BGZF format
     *
     * @param fn    file name
     * @return      1 if _fn_ is BGZF; 0 if not or on I/O error
     */
    static int bgzf_is_bgzf(const char *fn);

    /*********************
     * Advanced routines *
     *********************/

    /**
     * Set the cache size. Only effective when compiled with -DBGZF_CACHE.
     *
     * @param fp    BGZF file handler
     * @param size  size of cache in bytes; 0 to disable caching (default)
     */
    static void bgzf_set_cache_size(BGZF *fp, int size);

    /**
     * Flush the file if the remaining buffer size is smaller than _size_
     */
    static int bgzf_flush_try(BGZF *fp, ssize_t size);

    /**
     * Read one byte from a BGZF file. It is faster than bgzf_read()
     * @param fp     BGZF file handler
     * @return       byte read; -1 on end-of-file or error
     */
    static int bgzf_getc(BGZF *fp);

    /**
     * Read one line from a BGZF file. It is faster than bgzf_getc()
     *
     * @param fp     BGZF file handler
     * @param delim  delimitor
     * @param str    string to write to; must be initialized
     * @return       length of the string; 0 on end-of-file; negative on error
     */
    static int bgzf_getline(BGZF *fp, int delim, kstring_t *str);

    /**
     * Read the next BGZF block.
     */
    static int bgzf_read_block(BGZF *fp);

#ifdef BGZF_MT
    /**
     * Enable multi-threading (only effective on writing)
     *
     * @param fp          BGZF file handler; must be opened for writing
     * @param n_threads   #threads used for writing
     * @param n_sub_blks  #blocks processed by each thread; a value 64-256 is recommended
     */
    static int bgzf_mt(BGZF *fp, int n_threads, int n_sub_blks);
#endif

#ifdef __cplusplus
}
#endif

/* =========================================================================
 * Implementation (header-only; all symbols have internal linkage)
 * ========================================================================= */

/* ---- platform I/O abstraction ------------------------------------------ */
#ifdef _USE_KNETFILE
#include "knetfile.h"
typedef knetFile *_bgzf_file_t;
#define _bgzf_open(fn, mode) knet_open((fn), (mode))
#define _bgzf_dopen(fd, mode) knet_dopen((fd), (mode))
#define _bgzf_close(fp) knet_close((knetFile *)(fp))
#define _bgzf_fileno(fp) (((knetFile *)(fp))->fd)
#define _bgzf_tell(fp) knet_tell((knetFile *)(fp))
#define _bgzf_seek(fp, offset, whence) knet_seek((knetFile *)(fp), (offset), (whence))
#define _bgzf_read(fp, buf, len) knet_read((knetFile *)(fp), (buf), (len))
#define _bgzf_write(fp, buf, len) knet_write((knetFile *)(fp), (buf), (len))
#else // ~defined(_USE_KNETFILE)
#if defined(_WIN32) || defined(_MSC_VER)
#define ftello(fp) ftell((FILE *)(fp))
#define fseeko(fp, offset, whence) fseek((FILE *)(fp), (offset), (whence))
#else // ~defined(_WIN32)
// Suppress extern declarations in C++ -- they are in <stdio.h>/<unistd.h>
#ifndef __cplusplus
extern off_t ftello(FILE *stream);
extern int fseeko(FILE *stream, off_t offset, int whence);
#endif
#endif // ~defined(_WIN32)
typedef FILE *_bgzf_file_t;
#define _bgzf_open(fn, mode) fopen((fn), (mode))
#define _bgzf_dopen(fd, mode) fdopen(fd, (mode))
#define _bgzf_close(fp) fclose((FILE *)(fp))
#define _bgzf_fileno(fp) fileno((FILE *)(fp))
#define _bgzf_tell(fp) ftello((FILE *)(fp))
#define _bgzf_seek(fp, offset, whence) fseeko((FILE *)(fp), (offset), (whence))
#define _bgzf_read(fp, buf, len) fread((buf), 1, (len), (FILE *)(fp))
#define _bgzf_write(fp, buf, len) fwrite((buf), 1, (len), (FILE *)(fp))
#endif // ~defined(_USE_KNETFILE)

#define BLOCK_HEADER_LENGTH 18
#define BLOCK_FOOTER_LENGTH 8

/* BGZF/GZIP header (specialised from RFC 1952; little endian):
 +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
 | 31|139|  8|  4|              0|  0|255|      6| 66| 67|      2|BLK_LEN|
 +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
*/
static const uint8_t g_magic[19] = "\037\213\010\4\0\0\0\0\0\377\6\0\102\103\2\0\0\0";

#ifdef BGZF_CACHE
typedef struct
{
    int size;
    uint8_t *block;
    int64_t end_offset;
} cache_t;
#include "khash.h"
KHASH_MAP_INIT_INT64(cache, cache_t)
#endif

static inline int ed_is_big(void)
{
    long one = 1;
    return !(*((char *)(&one)));
}

static inline void packInt16(uint8_t *buffer, uint16_t value)
{
    buffer[0] = value;
    buffer[1] = value >> 8;
}

static inline int unpackInt16(const uint8_t *buffer)
{
    return buffer[0] | buffer[1] << 8;
}

static inline void packInt32(uint8_t *buffer, uint32_t value)
{
    buffer[0] = value;
    buffer[1] = value >> 8;
    buffer[2] = value >> 16;
    buffer[3] = value >> 24;
}

static BGZF *bgzf_read_init(void)
{
    BGZF *fp;
    fp = (BGZF *)calloc(1, sizeof(BGZF));
    fp->is_write = 0;
    fp->uncompressed_block = malloc(BGZF_MAX_BLOCK_SIZE);
    fp->compressed_block = malloc(BGZF_MAX_BLOCK_SIZE);
#ifdef BGZF_CACHE
    fp->cache = kh_init(cache);
#endif
    return fp;
}

static BGZF *bgzf_write_init(int compress_level) // compress_level==-1 for the default level
{
    BGZF *fp;
    fp = (BGZF *)calloc(1, sizeof(BGZF));
    fp->is_write = 1;
    fp->uncompressed_block = malloc(BGZF_MAX_BLOCK_SIZE);
    fp->compressed_block = malloc(BGZF_MAX_BLOCK_SIZE);
    fp->compress_level = compress_level < 0 ? Z_DEFAULT_COMPRESSION : compress_level;
    if (fp->compress_level > 9)
        fp->compress_level = Z_DEFAULT_COMPRESSION;
    return fp;
}
// get the compress level from the mode string
static int mode2level(const char *mode)
{
    int i, compress_level = -1;
    for (i = 0; mode[i]; ++i)
        if (mode[i] >= '0' && mode[i] <= '9')
            break;
    if (mode[i])
        compress_level = (int)mode[i] - '0';
    if (strchr(mode, 'u'))
        compress_level = 0;
    return compress_level;
}

static BGZF *bgzf_open(const char *path, const char *mode)
{
    BGZF *fp = 0;
    assert(compressBound(BGZF_BLOCK_SIZE) < BGZF_MAX_BLOCK_SIZE);
    if (strchr(mode, 'r') || strchr(mode, 'R'))
    {
        _bgzf_file_t fpr;
        if ((fpr = _bgzf_open(path, "r")) == 0)
            return 0;
        fp = bgzf_read_init();
        fp->fp = fpr;
    }
    else if (strchr(mode, 'w') || strchr(mode, 'W'))
    {
        FILE *fpw;
        if ((fpw = fopen(path, "w")) == 0)
            return 0;
        fp = bgzf_write_init(mode2level(mode));
        fp->fp = fpw;
    }
    fp->is_be = ed_is_big();
    return fp;
}

static BGZF *bgzf_dopen(int fd, const char *mode)
{
    BGZF *fp = 0;
    assert(compressBound(BGZF_BLOCK_SIZE) < BGZF_MAX_BLOCK_SIZE);
    if (strchr(mode, 'r') || strchr(mode, 'R'))
    {
        _bgzf_file_t fpr;
        if ((fpr = _bgzf_dopen(fd, "r")) == 0)
            return 0;
        fp = bgzf_read_init();
        fp->fp = fpr;
    }
    else if (strchr(mode, 'w') || strchr(mode, 'W'))
    {
        FILE *fpw;
        if ((fpw = fdopen(fd, "w")) == 0)
            return 0;
        fp = bgzf_write_init(mode2level(mode));
        fp->fp = fpw;
    }
    fp->is_be = ed_is_big();
    return fp;
}

static int bgzf_compress(void *_dst, int *dlen, void *src, int slen, int level)
{
    uint32_t crc;
    z_stream zs;
    uint8_t *dst = (uint8_t *)_dst;
    zs.zalloc = NULL;
    zs.zfree = NULL;
    zs.next_in = (Bytef *)src;
    zs.avail_in = slen;
    zs.next_out = dst + BLOCK_HEADER_LENGTH;
    zs.avail_out = *dlen - BLOCK_HEADER_LENGTH - BLOCK_FOOTER_LENGTH;
    if (deflateInit2(&zs, level, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY) != Z_OK)
        return -1;
    if (deflate(&zs, Z_FINISH) != Z_STREAM_END)
        return -1;
    if (deflateEnd(&zs) != Z_OK)
        return -1;
    *dlen = zs.total_out + BLOCK_HEADER_LENGTH + BLOCK_FOOTER_LENGTH;
    memcpy(dst, g_magic, BLOCK_HEADER_LENGTH);
    packInt16(&dst[16], *dlen - 1);
    crc = crc32(crc32(0L, NULL, 0L), (Bytef *)src, slen);
    packInt32((uint8_t *)&dst[*dlen - 8], crc);
    packInt32((uint8_t *)&dst[*dlen - 4], slen);
    return 0;
}

// Deflate the block in fp->uncompressed_block into fp->compressed_block. Also adds an extra field that stores the compressed block length.
static int deflate_block(BGZF *fp, int block_length)
{
    int comp_size = BGZF_MAX_BLOCK_SIZE;
    if (bgzf_compress(fp->compressed_block, &comp_size,
                      fp->uncompressed_block, block_length, fp->compress_level) != 0)
    {
        fp->errcode |= BGZF_ERR_ZLIB;
        return -1;
    }
    fp->block_offset = 0;
    return comp_size;
}

// Inflate the block in fp->compressed_block into fp->uncompressed_block
static int inflate_block(BGZF *fp, int block_length)
{
    z_stream zs;
    zs.zalloc = NULL;
    zs.zfree = NULL;
    zs.next_in = (Bytef *)fp->compressed_block + 18;
    zs.avail_in = block_length - 16;
    zs.next_out = (Bytef *)fp->uncompressed_block;
    zs.avail_out = BGZF_MAX_BLOCK_SIZE;
    if (inflateInit2(&zs, -15) != Z_OK)
    {
        fp->errcode |= BGZF_ERR_ZLIB;
        return -1;
    }
    if (inflate(&zs, Z_FINISH) != Z_STREAM_END)
    {
        inflateEnd(&zs);
        fp->errcode |= BGZF_ERR_ZLIB;
        return -1;
    }
    if (inflateEnd(&zs) != Z_OK)
    {
        fp->errcode |= BGZF_ERR_ZLIB;
        return -1;
    }
    return zs.total_out;
}

static int check_header(const uint8_t *header)
{   return
    (   header[0] == 31
    &&  header[1] == 139
    &&  header[2] == 8
    &&  (header[3] & 4) != 0
    &&  unpackInt16((uint8_t *)&header[10]) == 6
    &&  header[12] == 'B'
    &&  header[13] == 'C'
    &&  unpackInt16((uint8_t *)&header[14]) == 2
    );
}

#ifdef BGZF_CACHE
static void free_cache(BGZF *fp)
{
    khint_t k;
    khash_t(cache) *h = (khash_t(cache) *)fp->cache;
    if (fp->is_write)
        return;
    for (k = kh_begin(h); k < kh_end(h); ++k)
        if (kh_exist(h, k))
            free(kh_val(h, k).block);
    kh_destroy(cache, h);
}
static int load_block_from_cache(BGZF *fp, int64_t block_address)
{
    khint_t k;
    cache_t *p;
    khash_t(cache) *h = (khash_t(cache) *)fp->cache;
    k = kh_get(cache, h, block_address);
    if (k == kh_end(h))
        return 0;
    p = &kh_val(h, k);
    if (fp->block_length != 0)
        fp->block_offset = 0;
    fp->block_address = block_address;
    fp->block_length = p->size;
    memcpy(fp->uncompressed_block, p->block, BGZF_MAX_BLOCK_SIZE);
    _bgzf_seek((_bgzf_file_t)fp->fp, p->end_offset, SEEK_SET);
    return p->size;
}
static void cache_block(BGZF *fp, int size)
{
    int ret;
    khint_t k;
    cache_t *p;
    khash_t(cache) *h = (khash_t(cache) *)fp->cache;
    if (BGZF_MAX_BLOCK_SIZE >= fp->cache_size)
        return;
    if ((kh_size(h) + 1) * BGZF_MAX_BLOCK_SIZE > (uint32_t)fp->cache_size)
    {
        for (k = kh_begin(h); k < kh_end(h); ++k)
            if (kh_exist(h, k))
                break;
        if (k < kh_end(h))
        {
            free(kh_val(h, k).block);
            kh_del(cache, h, k);
        }
    }
    k = kh_put(cache, h, fp->block_address, &ret);
    if (ret == 0)
        return;
    p = &kh_val(h, k);
    p->size = fp->block_length;
    p->end_offset = fp->block_address + size;
    p->block = (uint8_t *)malloc(BGZF_MAX_BLOCK_SIZE);
    memcpy(kh_val(h, k).block, fp->uncompressed_block, BGZF_MAX_BLOCK_SIZE);
}
#else
static void free_cache(BGZF *fp)
{
    (void)fp;
}
static int load_block_from_cache(BGZF *fp, int64_t ba)
{
    (void)fp;
    (void)ba;
    return 0;
}
static void cache_block(BGZF *fp, int size)
{
    (void)fp;
    (void)size;
}
#endif // BGZF_CACHE

static int bgzf_read_block(BGZF *fp)
{
    uint8_t header[BLOCK_HEADER_LENGTH], *compressed_block;
    int count, size = 0, block_length, remaining;
    int64_t block_address;
    block_address = _bgzf_tell((_bgzf_file_t)fp->fp);
    if (fp->cache_size && load_block_from_cache(fp, block_address))
        return 0;
    count = _bgzf_read(fp->fp, header, sizeof(header));
    if (count == 0)
    {
        fp->block_length = 0;
        return 0;
    }
    if (count != sizeof(header) || !check_header(header))
    {
        fp->errcode |= BGZF_ERR_HEADER;
        return -1;
    }
    size = count;
    block_length = unpackInt16((uint8_t *)&header[16]) + 1;
    compressed_block = (uint8_t *)fp->compressed_block;
    memcpy(compressed_block, header, BLOCK_HEADER_LENGTH);
    remaining = block_length - BLOCK_HEADER_LENGTH;
    count = _bgzf_read(fp->fp, &compressed_block[BLOCK_HEADER_LENGTH], remaining);
    if (count != remaining)
    {
        fp->errcode |= BGZF_ERR_IO;
        return -1;
    }
    size += count;
    if ((count = inflate_block(fp, block_length)) < 0)
        return -1;
    if (fp->block_length != 0)
        fp->block_offset = 0;
    fp->block_address = block_address;
    fp->block_length = count;
    cache_block(fp, size);
    return 0;
}

static ssize_t bgzf_read(BGZF *fp, void *data, size_t length)
{
    ssize_t bytes_read = 0;
    uint8_t *output = (uint8_t *)data;
    if (length <= 0)
        return 0;
    assert(fp->is_write == 0);
    while (bytes_read < (ssize_t)length)
    {
        int copy_length, available = fp->block_length - fp->block_offset;
        uint8_t *buffer;
        if (available <= 0)
        {
            if (bgzf_read_block(fp) != 0)
                return -1;
            available = fp->block_length - fp->block_offset;
            if (available <= 0)
                break;
        }
        copy_length = (ssize_t)(length - bytes_read) < available
                          ? (int)(length - bytes_read)
                          : available;
        buffer = (uint8_t *)fp->uncompressed_block;
        memcpy(output, buffer + fp->block_offset, copy_length);
        fp->block_offset += copy_length;
        output += copy_length;
        bytes_read += copy_length;
    }
    if (fp->block_offset == fp->block_length)
    {
        fp->block_address = _bgzf_tell((_bgzf_file_t)fp->fp);
        fp->block_offset = fp->block_length = 0;
    }
    return bytes_read;
}

#ifdef BGZF_MT

typedef struct
{
    BGZF *fp;
    struct mtaux_t *mt;
    void *buf;
    int i, errcode, toproc;
} worker_t;

typedef struct mtaux_t
{
    int n_threads, n_blks, curr, done;
    volatile int proc_cnt;
    void **blk;
    int *len;
    worker_t *w;
    pthread_t *tid;
    pthread_mutex_t lock;
    pthread_cond_t cv;
} mtaux_t;

static int worker_aux(worker_t *w)
{
    int i, stop = 0;
    pthread_mutex_lock(&w->mt->lock);
    while (!w->toproc && !w->mt->done)
        pthread_cond_wait(&w->mt->cv, &w->mt->lock);
    if (w->mt->done)
        stop = 1;
    w->toproc = 0;
    pthread_mutex_unlock(&w->mt->lock);
    if (stop)
        return 1;
    w->errcode = 0;
    for (i = w->i; i < w->mt->curr; i += w->mt->n_threads)
    {
        int clen = BGZF_MAX_BLOCK_SIZE;
        if (bgzf_compress(w->buf, &clen, w->mt->blk[i], w->mt->len[i], w->fp->compress_level) != 0)
            w->errcode |= BGZF_ERR_ZLIB;
        memcpy(w->mt->blk[i], w->buf, clen);
        w->mt->len[i] = clen;
    }
    __sync_fetch_and_add(&w->mt->proc_cnt, 1);
    return 0;
}

static void *mt_worker(void *data)
{
    while (worker_aux((worker_t *)data) == 0)
        ;
    return 0;
}

static int bgzf_mt(BGZF *fp, int n_threads, int n_sub_blks)
{
    int i;
    mtaux_t *mt;
    pthread_attr_t attr;
    if (!fp->is_write || fp->mt || n_threads <= 1)
        return -1;
    mt = (mtaux_t *)calloc(1, sizeof(mtaux_t));
    mt->n_threads = n_threads;
    mt->n_blks = n_threads * n_sub_blks;
    mt->len = (int *)calloc(mt->n_blks, sizeof(int));
    mt->blk = (void **)calloc(mt->n_blks, sizeof(void *));
    for (i = 0; i < mt->n_blks; ++i)
        mt->blk[i] = malloc(BGZF_MAX_BLOCK_SIZE);
    mt->tid = (pthread_t *)calloc(mt->n_threads, sizeof(pthread_t));
    mt->w = (worker_t *)calloc(mt->n_threads, sizeof(worker_t));
    for (i = 0; i < mt->n_threads; ++i)
    {
        mt->w[i].i = i;
        mt->w[i].mt = mt;
        mt->w[i].fp = fp;
        mt->w[i].buf = malloc(BGZF_MAX_BLOCK_SIZE);
    }
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_mutex_init(&mt->lock, 0);
    pthread_cond_init(&mt->cv, 0);
    for (i = 1; i < mt->n_threads; ++i)
        pthread_create(&mt->tid[i], &attr, mt_worker, &mt->w[i]);
    fp->mt = mt;
    return 0;
}

static void mt_destroy(mtaux_t *mt)
{
    int i;
    pthread_mutex_lock(&mt->lock);
    mt->done = 1;
    mt->proc_cnt = 0;
    pthread_cond_broadcast(&mt->cv);
    pthread_mutex_unlock(&mt->lock);
    for (i = 1; i < mt->n_threads; ++i)
        pthread_join(mt->tid[i], 0);
    for (i = 0; i < mt->n_blks; ++i)
        free(mt->blk[i]);
    for (i = 0; i < mt->n_threads; ++i)
        free(mt->w[i].buf);
    free(mt->blk);
    free(mt->len);
    free(mt->w);
    free(mt->tid);
    pthread_cond_destroy(&mt->cv);
    pthread_mutex_destroy(&mt->lock);
    free(mt);
}

static void mt_queue(BGZF *fp)
{
    mtaux_t *mt = (mtaux_t *)fp->mt;
    assert(mt->curr < mt->n_blks);
    memcpy(mt->blk[mt->curr], fp->uncompressed_block, fp->block_offset);
    mt->len[mt->curr] = fp->block_offset;
    fp->block_offset = 0;
    ++mt->curr;
}

static int mt_flush(BGZF *fp)
{
    int i;
    mtaux_t *mt = (mtaux_t *)fp->mt;
    if (fp->block_offset)
        mt_queue(fp);
    pthread_mutex_lock(&mt->lock);
    for (i = 0; i < mt->n_threads; ++i)
        mt->w[i].toproc = 1;
    mt->proc_cnt = 0;
    pthread_cond_broadcast(&mt->cv);
    pthread_mutex_unlock(&mt->lock);
    worker_aux(&mt->w[0]);
    while (mt->proc_cnt < mt->n_threads)
        ;
    for (i = 0; i < mt->n_threads; ++i)
        fp->errcode |= mt->w[i].errcode;
    for (i = 0; i < mt->curr; ++i)
        if (fwrite(mt->blk[i], 1, mt->len[i], (FILE *)fp->fp) != (size_t)mt->len[i])
            fp->errcode |= BGZF_ERR_IO;
    mt->curr = 0;
    return 0;
}

static int mt_lazy_flush(BGZF *fp)
{
    mtaux_t *mt = (mtaux_t *)fp->mt;
    if (fp->block_offset)
        mt_queue(fp);
    if (mt->curr == mt->n_blks)
        return mt_flush(fp);
    return -1;
}

static ssize_t mt_write(BGZF *fp, const void *data, size_t length)
{
    const uint8_t *input = (const uint8_t *)data;
    ssize_t rest = length;
    while (rest)
    {
        int copy_length = BGZF_BLOCK_SIZE - fp->block_offset < rest
                              ? BGZF_BLOCK_SIZE - fp->block_offset
                              : (int)rest;
        memcpy((uint8_t *)fp->uncompressed_block + fp->block_offset, input, copy_length);
        fp->block_offset += copy_length;
        input += copy_length;
        rest -= copy_length;
        if (fp->block_offset == BGZF_BLOCK_SIZE)
            mt_lazy_flush(fp);
    }
    return length - rest;
}

#endif // ~ #ifdef BGZF_MT

static int bgzf_flush(BGZF *fp)
{
    if (!fp->is_write)
        return 0;
#ifdef BGZF_MT
    if (fp->mt)
        return mt_flush(fp);
#endif
    while (fp->block_offset > 0)
    {
        int block_length = deflate_block(fp, fp->block_offset);
        if (block_length < 0)
            return -1;
        if (fwrite(fp->compressed_block, 1, block_length, (FILE *)fp->fp) != (size_t)block_length)
        {
            fp->errcode |= BGZF_ERR_IO;
            return -1;
        }
        fp->block_address += block_length;
    }
    return 0;
}

static int bgzf_flush_try(BGZF *fp, ssize_t size)
{
    if (fp->block_offset + size > BGZF_BLOCK_SIZE)
    {
#ifdef BGZF_MT
        if (fp->mt)
            return mt_lazy_flush(fp);
        else
            return bgzf_flush(fp);
#else
        return bgzf_flush(fp);
#endif
    }
    return -1;
}

static ssize_t bgzf_write(BGZF *fp, const void *data, size_t length)
{
    const uint8_t *input = (const uint8_t *)data;
    int block_length = BGZF_BLOCK_SIZE, bytes_written = 0;
    assert(fp->is_write);
#ifdef BGZF_MT
    if (fp->mt)
        return mt_write(fp, data, length);
#endif
    while (bytes_written < (int)length)
    {
        uint8_t *buffer = (uint8_t *)fp->uncompressed_block;
        int copy_length = block_length - fp->block_offset < (int)length - bytes_written
                              ? block_length - fp->block_offset
                              : (int)length - bytes_written;
        memcpy(buffer + fp->block_offset, input, copy_length);
        fp->block_offset += copy_length;
        input += copy_length;
        bytes_written += copy_length;
        if (fp->block_offset == block_length && bgzf_flush(fp))
            break;
    }
    return bytes_written;
}

static int bgzf_close(BGZF *fp)
{
    int ret, block_length;
    if (fp == 0)
        return -1;
    if (fp->is_write)
    {
        if (bgzf_flush(fp) != 0)
            return -1;
        fp->compress_level = -1;
        block_length = deflate_block(fp, 0);
        fwrite(fp->compressed_block, 1, block_length, (FILE *)fp->fp);
        if (fflush((FILE *)fp->fp) != 0)
        {
            fp->errcode |= BGZF_ERR_IO;
            return -1;
        }
#ifdef BGZF_MT
        if (fp->mt)
            mt_destroy((mtaux_t *)fp->mt);
#endif
    }
    ret = fp->is_write ? fclose((FILE *)fp->fp) : _bgzf_close(fp->fp);
    if (ret != 0)
        return -1;
    free(fp->uncompressed_block);
    free(fp->compressed_block);
    free_cache(fp);
    free(fp);
    return 0;
}

static void bgzf_set_cache_size(BGZF *fp, int cache_size)
{
    if (fp)
        fp->cache_size = cache_size;
}

static int bgzf_check_EOF(BGZF *fp)
{
    uint8_t buf[28];
    off_t offset = _bgzf_tell((_bgzf_file_t)fp->fp);
    if (_bgzf_seek(fp->fp, -28, SEEK_END) < 0)
        return 0;
    std::ignore = _bgzf_read(fp->fp, buf, 28);
    _bgzf_seek(fp->fp, offset, SEEK_SET);
    return (memcmp("\037\213\010\4\0\0\0\0\0\377\6\0\102\103\2\0\033\0\3\0\0\0\0\0\0\0\0\0", buf, 28) == 0) ? 1 : 0;
}

static int64_t bgzf_seek(BGZF *fp, int64_t pos, int where)
{
    int block_offset;
    int64_t block_address;
    if (fp->is_write || where != SEEK_SET)
    {
        fp->errcode |= BGZF_ERR_MISUSE;
        return -1;
    }
    block_offset = pos & 0xFFFF;
    block_address = pos >> 16;
    if (_bgzf_seek(fp->fp, block_address, SEEK_SET) < 0)
    {
        fp->errcode |= BGZF_ERR_IO;
        return -1;
    }
    fp->block_length = 0;
    fp->block_address = block_address;
    fp->block_offset = block_offset;
    return 0;
}

static int bgzf_is_bgzf(const char *fn)
{
    uint8_t buf[16];
    int n;
    _bgzf_file_t fp;
    if ((fp = _bgzf_open(fn, "r")) == 0)
        return 0;
    n = (int)_bgzf_read(fp, buf, 16);
    _bgzf_close(fp);
    if (n != 16)
        return 0;
    return memcmp(g_magic, buf, 16) == 0 ? 1 : 0;
}

static int bgzf_getc(BGZF *fp)
{
    int c;
    if (fp->block_offset >= fp->block_length)
    {
        if (bgzf_read_block(fp) != 0)
            return -2;
        if (fp->block_length == 0)
            return -1;
    }
    c = ((unsigned char *)fp->uncompressed_block)[fp->block_offset++];
    if (fp->block_offset == fp->block_length)
    {
        fp->block_address = _bgzf_tell((_bgzf_file_t)fp->fp);
        fp->block_offset = 0;
        fp->block_length = 0;
    }
    return c;
}

static int bgzf_getline(BGZF *fp, int delim, kstring_t *str)
{
    int l, state = 0;
    unsigned char *buf = (unsigned char *)fp->uncompressed_block;
    str->l = 0;
    do
    {
        if (fp->block_offset >= fp->block_length)
        {
            if (bgzf_read_block(fp) != 0)
            {
                state = -2;
                break;
            }
            if (fp->block_length == 0)
            {
                state = -1;
                break;
            }
        }
        for (l = fp->block_offset; l < fp->block_length && buf[l] != delim; ++l)
            ;
        if (l < fp->block_length)
            state = 1;
        l -= fp->block_offset;
        if (str->l + l + 1 >= str->m)
        {
            str->m = str->l + l + 2;
            kroundup32(str->m);
            str->s = (char *)realloc(str->s, str->m);
        }
        memcpy(str->s + str->l, buf + fp->block_offset, l);
        str->l += l;
        fp->block_offset += l + 1;
        if (fp->block_offset >= fp->block_length)
        {
            fp->block_address = _bgzf_tell((_bgzf_file_t)fp->fp);
            fp->block_offset = 0;
            fp->block_length = 0;
        }
    } while (state == 0);
    if (str->l == 0 && state < 0)
        return state;
    str->s[str->l] = 0;
    return str->l;
}

#endif // __BGZF_H
