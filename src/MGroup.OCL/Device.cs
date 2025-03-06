using CASS.OpenCL;
using CASS.Types;

namespace MGroup.OCL
{
	public class Device
	{
		public readonly CLDeviceID deviceId; // Actually this is the only useful field used in OpenCL. All others are info for platform selection.
		public readonly String name;
		public readonly String vendor;
		public readonly String version;
		public readonly String extensions;
		public readonly String profile;
		public readonly CLDeviceType type;
		public readonly uint architectureBits;
		public readonly CLBool availableDevice;
		public readonly CLBool availableCompiler;
		public readonly String versionDriver;
		public readonly CLBool littleEndian;
		public readonly CLDeviceExecCapabilities executionCapabilities;
		public readonly ulong memSizeGlobal;
		public readonly ulong memSizeLocal;
		public readonly CLDeviceLocalMemType memTypeLocal;
		public readonly uint computeUnitsMax;
		public readonly ulong memSizeAllocMax;
		public readonly SizeT workgroupSizeMax;
		public readonly uint workitemDimensionsMax;
		public readonly SizeT[] workItemSizes;
		public readonly String versionOpenCL_C;
		public readonly uint vendorId;

		internal Device(CLDeviceID deviceId)
		{
			this.deviceId = deviceId;
			name = (String)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.Name);
			vendor = (String)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.Vendor);
			version = (String)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.Version);
			extensions = (String)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.Extensions);
			profile = (String)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.Profile);
			type = (CLDeviceType)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.Type);
			architectureBits = (uint)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.AddressBits);
			availableDevice = (CLBool)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.Available);
			availableCompiler = (CLBool)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.CompilerAvailable);
			versionDriver = (String)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.DriverVersion);
			littleEndian = (CLBool)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.EndianLittle);
			executionCapabilities = (CLDeviceExecCapabilities)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.ExecutionCapabilities);
			memSizeGlobal = (ulong)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.GlobalMemSize);
			memSizeLocal = (ulong)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.LocalMemSize);
			memTypeLocal = (CLDeviceLocalMemType)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.LocalMemType);
			computeUnitsMax = (uint)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.MaxComputeUnits);
			memSizeAllocMax = (ulong)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.MaxMemAllocSize);
			workgroupSizeMax = (SizeT)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.MaxWorkGroupSize);
			workitemDimensionsMax = (uint)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.MaxWorkItemDimensions);
			workItemSizes = (SizeT[])OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.MaxWorkItemSizes);
			versionOpenCL_C = (String)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.OpenCLCVersion);
			vendorId = (uint)OpenCL.GetDeviceInfo(deviceId, CLDeviceInfo.VendorID);
		}
	}
}
