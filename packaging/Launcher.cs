using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows.Forms;

// No shell command construction: quote each Windows argv element independently.
internal static class Launcher {
    private static string Quote(string value) {
        return "\"" + Regex.Replace(value, "(\\\\*)\"", "$1$1\\\"")
            .TrimEnd('\\') + new string('\\', value.Reverse().TakeWhile(c => c == '\\').Count() * 2) + "\"";
    }
    [STAThread]
    private static int Main(string[] args) {
        try {
            string root = AppDomain.CurrentDomain.BaseDirectory;
            var start = new ProcessStartInfo {
                FileName = Path.Combine(root, "runtime", "python", "pythonw.exe"),
                Arguments = "-I -B -X utf8 -m blab.desktop_runtime " + String.Join(" ", args.Select(Quote)),
                WorkingDirectory = root,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            using (var process = Process.Start(start)) {
                process.WaitForExit();
                return process.ExitCode;
            }
        } catch (Exception error) {
            MessageBox.Show(error.Message, "Boundary Lab could not start", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}
