using CSparse.Factorization;
using MGroup.Constitutive.Structural;
using MGroup.Constitutive.Structural.BoundaryConditions;
using MGroup.FEM.Structural.Tests.Plates.Commons;
using MGroup.FEM.Structural.Tests.Plates.FemExtentions;
using MGroup.FEM.Structural.Tests.Plates.Plotting;
using MGroup.LinearAlgebra.Vectors;
using MGroup.MSolve.Discretization.Entities;
using MGroup.MSolve.Discretization.Meshes.Structured;
using MGroup.MSolve.Solution;
using MGroup.MSolve.Solution.AlgebraicModel;
using MGroup.MSolve.Solution.LinearSystem;
using MGroup.NumericalAnalyzers;
using MGroup.Solvers.Direct;
using Xunit;

namespace MGroup.FEM.Structural.Tests.Plates
{
	public static class ContinuousPlates4Test
	{
		private const int subdomainID = 0;

		[Fact]
		public static void Run()
		{
			(Model model, UniformCartesianMesh2D mesh) = CreateModel();
			(IGlobalVector solution, IAlgebraicModel algebraicModel) = SolveModel(model);

			string outputFolder = "C:\\1";
			var plotter = new PlateStructurePlotter(model, algebraicModel, mesh, outputFolder);

			// Plot nodal displacements
			plotter.PlotDisplacementsAtNodes(solution);

			// Plot over a very fine grid
			int ratio = 10;
			int[] plotElements = new int[] { mesh.NumElements[0] * ratio, mesh.NumElements[1] * ratio };
			var plotGrid = new UniformCartesianMesh2D.Builder(mesh.MinCoordinates, mesh.MaxCoordinates, plotElements)
				.BuildMesh();
			//plotter.PlotDisplacementsAtGridPoints(plotGrid, solution);

			// Plot section moments
			plotter.PlotMomentsAtGridPoints(plotGrid, solution);
		}

		public static (Model model, UniformCartesianMesh2D mesh) CreateModel()
		{
			// Problem properties
			double Lx = 3.0;
			double Ly = 2.0;
			double thickness = 1; //0.3
			double E = 52416; //2E6
			double v = 0.3; //0.25
			int numElemX = 48;
			int numElemY = 32;
			double q = -10;

			// Mesh
			var mesh = new UniformCartesianMesh2D.Builder(
				new double[] {0, 0}, new double[] {Lx, Ly }, new int[] {numElemX, numElemY}).BuildMesh();

			// Nodes
			var model = new Model();
			model.SubdomainsDictionary[subdomainID] = new Subdomain(subdomainID);
			for (int n = 0; n < mesh.NumNodesTotal; n++)
			{
				double[] coords = mesh.GetNodeCoordinates(mesh.GetNodeIdx(n));
				model.NodesDictionary[n] = new Node(n, coords[0], coords[1]);
			}

			// Elements
			var elements = new List<PlateElementRectangle4Nodes3Dofs>();
			for (int e = 0; e < mesh.NumElementsTotal; e++)
			{
				int[] nodeIDs = mesh.GetElementConnectivity(e);
				var nodes = new INode[nodeIDs.Length];
				for (int n = 0; n <  nodeIDs.Length; n++)
				{
					nodes[n] = model.NodesDictionary[nodeIDs[n]];
				}
				var element = new PlateElementRectangle4Nodes3Dofs(nodes, E, v, thickness);
				model.ElementsDictionary[e] = element;
				model.SubdomainsDictionary[subdomainID].Elements.Add(element);
				elements.Add(element);
			}

			// Boundary conditions
			double distanceTol = 0.1 * Math.Min(Lx / (numElemX - 1), Ly / (numElemY - 1));
			var nodeLocator = new NodeLocator(model, distanceTol);

			// Supports
			var supports = new List<NodalDisplacement>();
			var supportedNodes = new HashSet<INode>();
			supportedNodes.UnionWith(nodeLocator.FindNodesWithX(0.0));
			supportedNodes.UnionWith(nodeLocator.FindNodesWithX(Lx / 2));
			supportedNodes.UnionWith(nodeLocator.FindNodesWithY(0.0));
			supportedNodes.UnionWith(nodeLocator.FindNodesWithY(Ly / 2));
			supportedNodes.UnionWith(nodeLocator.FindNodesWithY(Ly));
			foreach (INode node in supportedNodes.OrderBy(node => node.ID))
			{
				supports.Add(new NodalDisplacement(node, StructuralDof.TranslationZ, 0.0));
			}

			// Loads
			List<NodalLoad> nodalLoads = BodyForcesConverter.ApplyUniformLoadOnAllPlateElements(model, q, supports);
			model.BoundaryConditions.Add(new StructuralBoundaryConditionSet(supports, nodalLoads));

			return (model, mesh);
		}

		private static (IGlobalVector solution, IAlgebraicModel algebraicModel) SolveModel(Model model)
		{
			var solverFactory = new LdlSkylineSolver.Factory();
			var algebraicModel = solverFactory.BuildAlgebraicModel(model);
			ISolver solver = solverFactory.BuildSolver(algebraicModel);
			var problem = new ProblemStructural(model, algebraicModel);

			var linearAnalyzer = new LinearAnalyzer(algebraicModel, solver, problem);
			var staticAnalyzer = new StaticAnalyzer(algebraicModel, problem, linearAnalyzer);

			staticAnalyzer.Initialize();
			staticAnalyzer.Solve();

			return (solver.LinearSystem.Solution, algebraicModel);
		}
	}
}
