using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MGroup.MSolve.Discretization.Meshes.Structured;

namespace MGroup.FEM.Structural.Tests.Plates.Plotting
{
	public static class VtkExporter
	{
		/// <summary>
		/// Writes a 2D line plot as VTK polydata (points + polyline)
		/// </summary>
		/// <param name="filename">VTK file path</param>
		/// <param name="coords">Horizontal coordinate array</param>
		/// <param name="values">Quantity of interest array</param>
		public static void Write2DLinePlot(string filename, double[] coords, double[] values)
		{
			if (coords.Length != values.Length)
				throw new ArgumentException("X and Y must have the same length");

			int nPoints = coords.Length;

			using (StreamWriter sw = new StreamWriter(filename))
			{
				// Header
				sw.WriteLine("# vtk DataFile Version 3.0");
				sw.WriteLine("2D line plot");
				sw.WriteLine("ASCII");
				sw.WriteLine("DATASET POLYDATA");

				// Points
				sw.WriteLine($"POINTS {nPoints} float");
				for (int i = 0; i < nPoints; i++)
				{
					sw.WriteLine($"{coords[i]} {values[i]} 0.0");
				}

				// Polyline connecting the points
				sw.WriteLine($"LINES 1 {nPoints + 1}");
				sw.Write($"{nPoints}");
				for (int i = 0; i < nPoints; i++)
				{
					sw.Write($" {i}");
				}
				sw.WriteLine();

				// Optionally, store scalar values at points (same as Y here)
				sw.WriteLine($"POINT_DATA {nPoints}");
				sw.WriteLine("SCALARS Quantity float 1");
				sw.WriteLine("LOOKUP_TABLE default");
				for (int i = 0; i < nPoints; i++)
				{
					sw.WriteLine(values[i]);
				}
			}
		}

		public static void WriteScalarField(string filename, UniformCartesianMesh2D mesh, double[] scalarValuesAtNodes)
		{
			if (mesh.NumNodesTotal != scalarValuesAtNodes.Length)
			{
				throw new ArgumentException("There must be as many values as there are nodes");
			}

			using (StreamWriter sw = new StreamWriter(filename))
			{
				sw.WriteLine("# vtk DataFile Version 3.0");
				sw.WriteLine("Structured surface with quad cells");
				sw.WriteLine("ASCII");
				sw.WriteLine("DATASET UNSTRUCTURED_GRID");

				// Points
				sw.WriteLine($"POINTS {mesh.NumNodesTotal} float");
				for (int i = 0; i < mesh.NumNodesTotal; i++)
				{
					double[] coords = mesh.GetNodeCoordinates(mesh.GetNodeIdx(i));
					sw.WriteLine($"{coords[0]} {coords[1]} 0.0");
				}

				// Cells
				sw.WriteLine($"CELLS {mesh.NumElementsTotal} {mesh.NumElementsTotal * 5}"); // 4 points + 1 for size
				for (int i = 0; i < mesh.NumElementsTotal; i++)
				{
					int[] nodes = mesh.GetElementConnectivity(i);
					sw.WriteLine($"4 {nodes[0]} {nodes[1]} {nodes[2]} {nodes[3]}");
				}

				// Cell types: 9 = quad
				sw.WriteLine($"CELL_TYPES {mesh.NumElementsTotal}");
				for (int i = 0; i < mesh.NumElementsTotal; i++)
				{
					sw.WriteLine("9");
				}

				// Point scalar field
				sw.WriteLine($"POINT_DATA {mesh.NumNodesTotal}");
				sw.WriteLine("SCALARS scalar float 1");
				sw.WriteLine("LOOKUP_TABLE default");
				for (int i = 0; i < mesh.NumNodesTotal; i++)
				{
					sw.WriteLine(scalarValuesAtNodes[i]);
				}
			}
		}

		public static void WriteScalarFields(string filename, UniformCartesianMesh2D mesh, Dictionary<string, double[]> valuesAtNodesPerScalarField)
		{
			using (StreamWriter sw = new StreamWriter(filename))
			{
				sw.WriteLine("# vtk DataFile Version 3.0");
				sw.WriteLine("Structured surface with quad cells");
				sw.WriteLine("ASCII");
				sw.WriteLine("DATASET UNSTRUCTURED_GRID");

				// Points
				sw.WriteLine($"POINTS {mesh.NumNodesTotal} float");
				for (int i = 0; i < mesh.NumNodesTotal; i++)
				{
					double[] coords = mesh.GetNodeCoordinates(mesh.GetNodeIdx(i));
					sw.WriteLine($"{coords[0]} {coords[1]} 0.0");
				}

				// Cells
				sw.WriteLine($"CELLS {mesh.NumElementsTotal} {mesh.NumElementsTotal * 5}"); // 4 points + 1 for size
				for (int i = 0; i < mesh.NumElementsTotal; i++)
				{
					int[] nodes = mesh.GetElementConnectivity(i);
					sw.WriteLine($"4 {nodes[0]} {nodes[1]} {nodes[2]} {nodes[3]}");
				}

				// Cell types: 9 = quad
				sw.WriteLine($"CELL_TYPES {mesh.NumElementsTotal}");
				for (int i = 0; i < mesh.NumElementsTotal; i++)
				{
					sw.WriteLine("9");
				}

				// Point scalar fields
				sw.WriteLine($"POINT_DATA {mesh.NumNodesTotal}");
				foreach (var field in valuesAtNodesPerScalarField)
				{
					string name = field.Key;
					double[] values = field.Value;

					if (mesh.NumNodesTotal != values.Length)
					{
						throw new ArgumentException($"Field {name} has {values.Length} values, while the mesh has {mesh.NumNodesTotal} nodes");
					}

					sw.WriteLine($"SCALARS {field.Key} float 1");
					sw.WriteLine("LOOKUP_TABLE default");
					for (int i = 0; i < mesh.NumNodesTotal; i++)
					{
						sw.WriteLine(values[i]);
					}
				}
			}
		}

		public static void WriteVectorField(string filename, UniformCartesianMesh2D mesh, List<double[]> vectorValuesAtNodes)
		{
			if (mesh.NumNodesTotal != vectorValuesAtNodes.Count)
			{
				throw new ArgumentException("There must be as many values as there are nodes");
			}

			using (StreamWriter sw = new StreamWriter(filename))
			{
				sw.WriteLine("# vtk DataFile Version 3.0");
				sw.WriteLine("Structured surface with quad cells");
				sw.WriteLine("ASCII");
				sw.WriteLine("DATASET UNSTRUCTURED_GRID");

				// Points
				sw.WriteLine($"POINTS {mesh.NumNodesTotal} float");
				for (int i = 0; i < mesh.NumNodesTotal; i++)
				{
					double[] coords = mesh.GetNodeCoordinates(mesh.GetNodeIdx(i));
					sw.WriteLine($"{coords[0]} {coords[1]} 0.0");
				}

				// Cells
				sw.WriteLine($"CELLS {mesh.NumElementsTotal} {mesh.NumElementsTotal * 5}"); // 4 points + 1 for size
				for (int i = 0; i < mesh.NumElementsTotal; i++)
				{
					int[] nodes = mesh.GetElementConnectivity(i);
					sw.WriteLine($"4 {nodes[0]} {nodes[1]} {nodes[2]} {nodes[3]}");
				}

				// Cell types: 9 = quad
				sw.WriteLine($"CELL_TYPES {mesh.NumElementsTotal}");
				for (int i = 0; i < mesh.NumElementsTotal; i++)
				{
					sw.WriteLine("9");
				}

				// Point vector field: displacement or velocity, etc.
				sw.WriteLine($"POINT_DATA {mesh.NumNodesTotal}");
				sw.WriteLine("VECTORS displacement float");
				for (int i = 0; i < mesh.NumNodesTotal; i++)
				{
					double[] v = vectorValuesAtNodes[i];
					sw.WriteLine($"{v[0]} {v[1]} {v[2]}");
				}
			}
		}
	}
}
