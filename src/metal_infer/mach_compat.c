// Thin C wrapper around mach task_info to avoid Zig 0.16 cImport issues
// with mach/message.h opaque union types.
#include <mach/mach.h>
#include <mach/task.h>
#include <stdint.h>

// Returns current process RSS in bytes, or 0 on failure.
uint64_t dmlx_get_rss_bytes(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                  (task_info_t)&info, &count);
    if (kr != KERN_SUCCESS) return 0;
    return (uint64_t)info.resident_size;
}

// Returns current process virtual size in bytes, or 0 on failure.
uint64_t dmlx_get_virtual_bytes(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                  (task_info_t)&info, &count);
    if (kr != KERN_SUCCESS) return 0;
    return (uint64_t)info.virtual_size;
}
