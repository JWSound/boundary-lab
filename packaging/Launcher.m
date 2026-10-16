#import <Cocoa/Cocoa.h>
#include <unistd.h>
#include <stdio.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        NSString *root = [[NSBundle mainBundle] resourcePath];
        NSString *python = [root stringByAppendingPathComponent:@"runtime/python/bin/python3"];
        char **args = calloc(argc + 7, sizeof(char *));
        args[0] = (char *)[python fileSystemRepresentation];
        args[1] = "-I";
        args[2] = "-B";
        args[3] = "-X";
        args[4] = "utf8";
        args[5] = "-m";
        args[6] = "blab.desktop_runtime";
        for (int i = 1; i < argc; ++i) args[i + 6] = argv[i];
        NSString *logs = [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Logs/Boundary Lab"];
        [[NSFileManager defaultManager] createDirectoryAtPath:logs withIntermediateDirectories:YES attributes:nil error:nil];
        const char *log = [[logs stringByAppendingPathComponent:@"launcher.log"] fileSystemRepresentation];
        freopen(log, "a", stdout);
        freopen(log, "a", stderr);
        execv(args[0], args);
        NSAlert *alert = [[NSAlert alloc] init];
        alert.messageText = @"Boundary Lab could not start";
        alert.informativeText = @"See ~/Library/Logs/Boundary Lab/launcher.log";
        [alert runModal];
        return 1;
    }
}
