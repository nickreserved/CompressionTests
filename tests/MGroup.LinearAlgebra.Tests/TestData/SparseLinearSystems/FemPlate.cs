using Compression.src.MGroup.Solvers.Multigrid;
using MGroup.Constitutive.Structural;
using MGroup.Constitutive.Structural.BoundaryConditions;
using MGroup.FEM.Structural.Tests.Plates.Commons;
using MGroup.FEM.Structural.Tests.Plates.FemExtentions;
using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Matrices.Builders;
using MGroup.LinearAlgebra.Vectors;
using MGroup.MSolve.Discretization.Entities;
using MGroup.MSolve.Discretization.Meshes.Structured;
using MGroup.MSolve.Solution;
using MGroup.NumericalAnalyzers;
using MGroup.Solvers.Iterative;
using MGroup.Solvers.LinearSystem;

namespace Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems
{
    public class FemPlate : IGeometricMultigridModel
    {
        /// <summary>
        /// Number of nodes per node are 1 displacement dofs per node and 2 rotation dofs per node.
        /// </summary>
        public int NumDofsPerNode { get;  } = 3;
        public UniformCartesianMesh2D Mesh { get; }
        IStructuredMesh IGeometricMultigridModel.Mesh { get => Mesh; }

        protected Model model;

        public int NumDofsAll { get; private set; }

        public int NumDofsFree { get; private set; }

        public double ElasticityModulus { get; set; } = 1.0;
        public double PoissonRatio { get; set; } = 0.3;
        public double DistributedLoad { get; set; } = 0;
        public double Thickness { get; } = 1;

        public (DokRowMajor A, Vector b) CreateLinearSystem()
        { 
            if (model == null) model = CreateModel(Mesh, Thickness, ElasticityModulus, PoissonRatio, DistributedLoad);

            var solverFactory = new PcgSolver.Factory();
            //var solverFactory = new LdlSkylineSolver.Factory();
            var algebraicModel = solverFactory.BuildAlgebraicModel(model);
            ISolver solver = solverFactory.BuildSolver(algebraicModel);
            var problem = new ProblemStructural(model, algebraicModel);
            var linearAnalyzer = new LinearAnalyzer(algebraicModel, solver, problem);
            var staticAnalyzer = new StaticAnalyzer(algebraicModel, problem, linearAnalyzer);
            staticAnalyzer.Initialize();
            //staticAnalyzer.Solve();

            CsrMatrix csrMatrix = algebraicModel.LinearSystem.Matrix.SingleMatrix;
            DokRowMajor dokMatrix = DokRowMajor.CreateFromSparseMatrix(csrMatrix);
            Vector rhsVector = ((GlobalVector) problem.GetRhs()).SingleVector;
            return (dokMatrix, rhsVector);
        }

        public int[] FindFreeDofs()
        {
            var freeDofs = new int[NumDofsFree];
            var supportedNodes = GetFixedNodes(Mesh, model);
            int curNodeId = 0;
            int curDofId = 0;
            int n = 0;
            foreach (INode node in supportedNodes.OrderBy(node => node.ID))
            {
                for (; curNodeId < node.ID; ++curNodeId)    // curNodeId < node.ID
                {
                    freeDofs[n++] = curDofId++;
                    freeDofs[n++] = curDofId++;
                    freeDofs[n++] = curDofId++;
                }
                                curDofId++;                 // curNodeId == node.ID, so TRANSLATE_Z is fixed and not stored in freeDofs
                freeDofs[n++] = curDofId++;
                freeDofs[n++] = curDofId++;
            }
            for (; curNodeId < Mesh.NumNodesTotal; ++curNodeId)        // curNodeId > last(node.ID)
            {
                freeDofs[n++] = curDofId++;
                freeDofs[n++] = curDofId++;
                freeDofs[n++] = curDofId++;
            }

            return freeDofs;
        }

        public IGeometricMultigridModel CreateCoarserModel()
        {
            int[] numElementsPerAxis = (int[])Mesh.NumElements.Clone();
            IGeometricMultigridModel.MakeCartesianCoarserElementsPerAxis(numElementsPerAxis);
            return new FemPlate(numElementsPerAxis, Mesh.MaxCoordinates.Zip(Mesh.MinCoordinates, (x, y) => x - y).Append(Thickness).ToArray());
        }

        /// <summary>
        /// Creates a cantilever Model.
        /// </summary>
        /// <param name="numElementsPerAxis">Array with 2 entries. The number of elements on cantilever's length (0) and height (1) axis.</param>
        /// <param name="lengthPerAxis">Array with 3 entries. The entries are cantilever's length (0), height (1), width (2).</param>
        public FemPlate(int[] numElementsPerAxis, double[] lengthPerAxis)
        {
            Mesh = new UniformCartesianMesh2D.Builder(new double[] { 0, 0 }, lengthPerAxis[..2], numElementsPerAxis).BuildMesh();
            Thickness = lengthPerAxis[2];
            NumDofsAll = NumDofsPerNode * Mesh.NumNodesTotal;
            NumDofsFree = NumDofsAll
                - 2 * Mesh.NumNodes[1]    // 2 rows with constant x has its TRANSLATE_Z (only) fixed
                - 3 * Mesh.NumNodes[0]    // 3 rows with constant y has its TRANSLATE_Z (only) fixed
                + 6;                      // 2*3 intersection points with constant y AND constant x counted twise
        }

        private static Model CreateModel(UniformCartesianMesh2D mesh, double Thickness,
                                                                              double ElasticityModulus, double PoissonRatio,
                                                                              double DistributedLoad)
        {
            const int subdomainID = 0;
            // Problem properties
            //double Lx = 3.0;
            //double Ly = 2.0;
            //double E = 52416; //2E6
            //double v = 0.3; //0.25
            //int numElemX = 48;
            //int numElemY = 32;
            //double q = -10; 

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
                for (int n = 0; n < nodeIDs.Length; n++)
                {
                    nodes[n] = model.NodesDictionary[nodeIDs[n]];
                }
                var element = new PlateElementRectangle4Nodes3Dofs(nodes, ElasticityModulus, PoissonRatio, Thickness);
                model.ElementsDictionary[e] = element;
                model.SubdomainsDictionary[subdomainID].Elements.Add(element);
                elements.Add(element);
            }

            // Supports
            var supportedNodes = GetFixedNodes(mesh, model);
            var supports = new List<NodalDisplacement>();
            foreach (INode node in supportedNodes.OrderBy(node => node.ID))
                supports.Add(new NodalDisplacement(node, StructuralDof.TranslationZ, 0.0));

            // Loads
            List<NodalLoad> nodalLoads = BodyForcesConverter.ApplyUniformLoadOnAllPlateElements(model, DistributedLoad, supports);
            model.BoundaryConditions.Add(new StructuralBoundaryConditionSet(supports, nodalLoads));

            return model;
        }

        private static HashSet<INode> GetFixedNodes(ICartesianMesh mesh, Model model)
        {
            double Lx = mesh.MaxCoordinates[0] - mesh.MinCoordinates[0];
            double Ly = mesh.MaxCoordinates[1] - mesh.MinCoordinates[1];
            // Boundary conditions
            double distanceTol = 0.1 * Math.Min(Lx / (mesh.NumElements[0] - 1), Ly / (mesh.NumElements[1] - 1));
            var nodeLocator = new NodeLocator(model, distanceTol);

            // Supports
            var supportedNodes = new HashSet<INode>();
            supportedNodes.UnionWith(nodeLocator.FindNodesWithX(0.0));
            supportedNodes.UnionWith(nodeLocator.FindNodesWithX(Lx / 2));
            supportedNodes.UnionWith(nodeLocator.FindNodesWithY(0.0));
            supportedNodes.UnionWith(nodeLocator.FindNodesWithY(Ly / 2));
            supportedNodes.UnionWith(nodeLocator.FindNodesWithY(Ly));

            return supportedNodes;
        }
    }
}
