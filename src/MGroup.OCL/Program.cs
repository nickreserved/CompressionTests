using CASS.OpenCL;
using CASS.Types;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;


namespace Compression.src.MGroup.OCL
{
    public class Program
    {
        /// <summary>
        /// Create a program, load a precompiled program or compile the source of a program.
        /// </summary>
        /// <param name="instance">OpenCL instance</param>
        /// <param name="programFile">Path and filename of source file of the program without extension ".cl"</param>
        /// <param name="compilerOptions">Compiler options. They take place only if precompiled program will not found. It becomes part of binary file name. Optional.</param>
        /// <returns>The OpenCL program</returns>
        public static CLProgram CreateProgram(OpenCL instance, String programFile, String? compilerOptions = null)
        {
            CLProgram program;
            // Because instance has only the deviceId, we must re-get the name of device
            String[] binaryFilename = new string[instance.Devices.Length];
            for (int i = 0; i < binaryFilename.Length; ++i)
            {
                String co = compilerOptions != null ? "." + compilerOptions : "";
                binaryFilename[i] = programFile + co + "." + ((String)OpenCL.GetDeviceInfo(instance.Devices[i], CLDeviceInfo.Name)).TrimEnd('\0') + ".bin";
            }

            try   // try to load a binary from a previous compiled source if exist - if not, or if it is wrong, fallback to source
            {
                byte[][] binary = new byte[instance.Devices.Length][];
                for (int i = 0; i < binary.Length; ++i)
                    binary[i] = File.ReadAllBytes(binaryFilename[i]);   // load can throw because binary for this specific device may be missed

                program = instance.CreateProgramWithBinary(binary);
                instance.BuildProgram(program, instance.Devices, null); // build can throw because binary can be from another device
            }
            catch
            {
                program = instance.CreateProgramWithSource(File.ReadAllText(programFile + ".cl"));
                const String BUILD_LOG_FILE = "OpenCLBuildingError.log";
                try { instance.BuildProgram(program, instance.Devices, compilerOptions); }
                catch
                {
                    File.WriteAllBytes(BUILD_LOG_FILE, Encoding.UTF8.GetBytes((String)
                            OpenCL.GetProgramBuildInfo(program, instance.Devices[0], CLProgramBuildInfo.Log)));
                    throw;
                }
                File.Delete(BUILD_LOG_FILE);
                // time to same compiled binaries
                byte[][] binary = (byte[][]) OpenCL.GetProgramInfo(program, CLProgramInfo.Binaries);
                for (int i = 0; i < binary.Length; ++i)
                    File.WriteAllBytes(binaryFilename[i], binary[i]);
            }
            return program;
        }
    }
}
