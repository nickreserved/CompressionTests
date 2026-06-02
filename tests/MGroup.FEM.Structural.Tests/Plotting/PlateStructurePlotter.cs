using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MGroup.Constitutive.Structural;
using MGroup.FEM.Structural.Tests.Plates.FemExtentions;
using MGroup.LinearAlgebra.Vectors;
using MGroup.MSolve.Discretization.Dofs;
using MGroup.MSolve.Discretization.Entities;
using MGroup.MSolve.Discretization.Meshes.Structured;
using MGroup.MSolve.Solution.AlgebraicModel;
using MGroup.MSolve.Solution.LinearSystem;
using MGroup.Solvers.Results;

namespace MGroup.FEM.Structural.Tests.Plates.Plotting
{
	public class PlateStructurePlotter
	{
		private readonly Model model;
		private readonly IAlgebraicModel algebraicModel;
		private readonly UniformCartesianMesh2D mesh;
		private readonly string directoryPath;

		public PlateStructurePlotter(Model model, IAlgebraicModel algebraicModel, UniformCartesianMesh2D mesh, string directoryPath)
		{
			this.model = model;
			this.algebraicModel = algebraicModel;
			this.mesh = mesh;
			this.directoryPath = directoryPath;
		}

		public void PlotDisplacementsAtNodes(IGlobalVector globalFreeDisplacements)
		{
			int numNodes = mesh.NumNodesTotal;
			var pointDisplacements = new List<double[]>(numNodes);
			for (int n = 0; n < numNodes; n++)
			{
				double[] globalCoords = mesh.GetNodeCoordinates(mesh.GetNodeIdx(n));
				INode node = model.NodesDictionary[n];
				double w;
				try
				{
					w = algebraicModel.ExtractSingleValue(globalFreeDisplacements, node, StructuralDof.TranslationZ);
				}
				catch (Exception ex) 
				{
					w = 0.0; // Constrained dof
				}
				pointDisplacements.Add(new double[] { 0, 0, w });
			}

			string file = Path.Combine(directoryPath, "displacements_at_nodes.vtk");
			VtkExporter.WriteVectorField(file, mesh, pointDisplacements);
		}

		public void PlotDisplacementsAtGridPoints(UniformCartesianMesh2D plotGrid, IGlobalVector globalFreeDisplacements)
		{
			int numPoints = plotGrid.NumNodesTotal;
			var pointDisplacements = new List<double[]>(numPoints);
			for (int p = 0; p < numPoints; p++)
			{
				double[] globalCoords = plotGrid.GetNodeCoordinates(plotGrid.GetNodeIdx(p));
				(int elementID, double[] localCoords) = GlobalToLocalCoords(globalCoords);
				var element = (PlateElementRectangle4Nodes3Dofs)model.ElementsDictionary[elementID];
				double[] displAtElementNodes = algebraicModel.ExtractElementVector(globalFreeDisplacements, element);
				double w = element.CalcVerticalDisplacementAt(localCoords, displAtElementNodes);
				pointDisplacements.Add(new double[] { 0, 0, w });
			}

			string file = Path.Combine(directoryPath, "displacements.vtk");
			VtkExporter.WriteVectorField(file, plotGrid, pointDisplacements);
		}

		public void PlotMomentsAtGridPoints(UniformCartesianMesh2D plotGrid, IGlobalVector globalFreeDisplacements)
		{
			int numPoints = plotGrid.NumNodesTotal;

			var momentsMx = new double[numPoints];
			var momentsMy = new double[numPoints];
			var momentsMxy = new double[numPoints];

			for (int p = 0; p < numPoints; p++)
			{
				double[] globalCoords = plotGrid.GetNodeCoordinates(plotGrid.GetNodeIdx(p));
				(int elementID, double[] localCoords) = GlobalToLocalCoords(globalCoords);
				var element = (PlateElementRectangle4Nodes3Dofs)model.ElementsDictionary[elementID];
				double[] displAtElementNodes = algebraicModel.ExtractElementVector(globalFreeDisplacements, element);

				double[] M = element.CalcSectionMomentsAt(localCoords, displAtElementNodes);
				momentsMx[p] = - M[0];
				momentsMy[p] = - M[1];
				momentsMxy[p] = - M[2];
			}

			var pointMoments = new Dictionary<string, double[]>(3);
			pointMoments["Mx"] = momentsMx;
			pointMoments["My"] = momentsMy;
			pointMoments["Mxy"] = momentsMxy;

			string file = Path.Combine(directoryPath, "section_moments.vtk");
			VtkExporter.WriteScalarFields(file, plotGrid, pointMoments);
		}

		private (int elementID, double[] localCoords) GlobalToLocalCoords(double[] globalCoords)
		{
			double x = globalCoords[0];
			double y = globalCoords[1];
			double xMin = mesh.MinCoordinates[0];
			double yMin = mesh.MinCoordinates[1];
			double dx = mesh.DistancesBetweenPoints[0];
			double dy = mesh.DistancesBetweenPoints[1];

			// Check that the coordinates are inside the plate
			double tol = 1E-3 * Math.Min(mesh.DistancesBetweenPoints[0], mesh.DistancesBetweenPoints[1]);
			if (x < xMin - tol || x > mesh.MaxCoordinates[0] + tol
				|| y < yMin - tol || y > mesh.MaxCoordinates[1] + tol)
			{
				throw new Exception("Point lies outside the FEM mesh");
			}

			// Find the element
			int ex = (int)Math.Floor((x - xMin) / dx);
			if (ex < 0)
			{
				ex = 0;
			}
			else if (ex >= mesh.NumElements[0])
			{
				ex = mesh.NumElements[0] - 1;
			}

			int ey = (int)Math.Floor((y - yMin) / dy);
			if (ey < 0)
			{
				ey = 0;
			}
			else if (ey >= mesh.NumElements[1])
			{
				ey = mesh.NumElements[1] - 1;
			}

			int elementID = mesh.GetElementID(new int[] { ex, ey });

			// Find the local coordinates of the point within that element
			double xCenter = xMin + ex * dx + dx / 2;
			double yCenter = yMin + ey * dy + dy / 2;
			double xLoc = x - xCenter;
			double yLoc = y - yCenter;

			return (elementID, new double[] { xLoc, yLoc });
		}
	}
}
