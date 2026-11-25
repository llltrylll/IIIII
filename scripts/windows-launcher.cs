using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class Program
{
    private const string Backend = "vulkan";

    [STAThread]
    private static int Main()
    {
        string folder = AppDomain.CurrentDomain.BaseDirectory;
        string application = Path.Combine(folder, "Material Painter.exe");

        if (!File.Exists(application))
        {
            MessageBox.Show(
                "Material Painter.exe was not found. Extract the complete Material Painter package before launching.",
                "Material Painter Vulkan launcher",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error);
            return 2;
        }

        try
        {
            var start = new ProcessStartInfo
            {
                FileName = application,
                WorkingDirectory = folder,
                UseShellExecute = false
            };
            start.EnvironmentVariables["WGPU_BACKEND"] = Backend;
            start.EnvironmentVariables["RUST_BACKTRACE"] = "1";
            Process.Start(start);
            return 0;
        }
        catch (Exception error)
        {
            MessageBox.Show(
                "Could not start Material Painter using Vulkan:\r\n\r\n" +
                error.Message,
                "Material Painter Vulkan launcher",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error);
            return 1;
        }
    }
}
