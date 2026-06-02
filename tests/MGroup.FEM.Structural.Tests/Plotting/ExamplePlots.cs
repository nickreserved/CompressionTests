using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MGroup.MSolve.Discretization.Meshes.Structured;
using Xunit;

namespace MGroup.FEM.Structural.Tests.Plates.Plotting
{
	public static class ExamplePlots
	{
		[Fact]
		public static void LinePlot()
		{
			int numPoints = 20;
			var x = new double[numPoints];
			var f = new double[numPoints];
			for (int i = 0; i < numPoints; i++)
			{
				x[i] = 0.1 * i;
				f[i] = x[i] * x[i];
			}
			string path = "C:\\1\\quadraticFuncDiagram.vtk";
			VtkExporter.Write2DLinePlot(path, x, f);
		}

		[Fact]
		public static void SurfacePlot()
		{
			var minCoords = new double[] { 0, 0 };
			var maxCoords = new double[] { 15, 10 };
			var numCells = new int[] { 15, 10 };

			var mesh = new UniformCartesianMesh2D.Builder(minCoords, maxCoords, numCells).BuildMesh();
			var f = new double[mesh.NumNodesTotal];
			var w = new List<double[]>(mesh.NumNodesTotal);
			for (int n = 0; n < mesh.NumNodesTotal; n++)
			{
				double[] x = mesh.GetNodeCoordinates(mesh.GetNodeIdx(n));
				//f[n] = Math.Sin(Math.PI * x[0]) * Math.Cos(Math.PI * x[1]);
				f[n] = x[1] * Math.Pow(x[0], 2);
				w.Add(new double[] { 0, 0, f[n] });
			}

			VtkExporter.WriteScalarField("C:\\1\\surface_scalar.vtk", mesh, f);
			VtkExporter.WriteVectorField("C:\\1\\surface_vector.vtk", mesh, w);
		}
	}
}
