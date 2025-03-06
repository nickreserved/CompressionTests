using CASS.OpenCL;

namespace MGroup.OCL
{
	public class Platform
	{
		public readonly CLPlatformID platformId; // Actually this is the only useful field used in OpenCL. All others are info for platform selection.
		public readonly String name;
		public readonly String vendor;
		public readonly String version;
		public readonly String extensions;
		public readonly String profile;
		private Device[]? devices;

		private Platform(CLPlatformID platformId)
		{
			this.platformId = platformId;
			name = (String)OpenCL.GetPlatformInfo(platformId, CLPlatformInfo.Name);
			vendor = (String)OpenCL.GetPlatformInfo(platformId, CLPlatformInfo.Vendor);
			version = (String)OpenCL.GetPlatformInfo(platformId, CLPlatformInfo.Version);
			extensions = (String)OpenCL.GetPlatformInfo(platformId, CLPlatformInfo.Extensions);
			profile = (String)OpenCL.GetPlatformInfo(platformId, CLPlatformInfo.Profile);
		}

		private static Platform[]? platforms;
		public static Platform[] GetPlatforms()
		{
			if (platforms != null) return platforms;

			CLPlatformID[] platformId = OpenCL.GetPlatforms();
			platforms = new Platform[platformId.Length];
			for (int i = 0; i < platforms.Length; ++i)
				platforms[i] = new Platform(platformId[i]);
			return platforms;
		}

		public Device[] GetDevices()
		{
			if (devices != null) return devices;

			CLDeviceID[] deviceId = OpenCL.GetDevices(platformId);
			devices = new Device[deviceId.Length];
			for (int i = 0; i < devices.Length; ++i)
				devices[i] = new Device(deviceId[i]);
			return devices;
		}

	}
}
