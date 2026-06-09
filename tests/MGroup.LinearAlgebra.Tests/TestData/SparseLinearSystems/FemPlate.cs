using Compression.src.MGroup.Solvers.Multigrid;
using MGroup.Constitutive.Structural;
using MGroup.Constitutive.Structural.BoundaryConditions;
using MGroup.FEM.Structural.Tests.Plates.Commons;
using MGroup.FEM.Structural.Tests.Plates.FemExtentions;
using MGroup.LinearAlgebra.Matrices.Builders;
using MGroup.LinearAlgebra.Vectors;
using MGroup.MSolve.Discretization.Entities;
using MGroup.MSolve.Discretization.Meshes.Structured;

namespace Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems
{
    public class FemPlate : IGeometricMultigridModel
    {
        public UniformCartesianMesh2D Mesh { get; }
        IStructuredMesh IStructuredModel.Mesh { get => Mesh; }

        public int NumDofsAll { get; private set; }

        public int NumDofsFree { get; private set; }

        public double ElasticityModulus { get; set; } = 1.0;
        public double PoissonRatio { get; set; } = 0.3;
        public double DistributedLoad { get; set; } = 1.0;
        public double Thickness { get; }

        public IGeometricMultigridModel CreateCoarserModel()
        {
            throw new NotImplementedException(); //TODO: it MUST be implemented
        }

        public (IGeometricMultigridModel coarserModel, DokRowMajor restrictionMatrix, DokRowMajor interpolationMatrix) CreateCoarserModelAndSmoothenerMatrices()
        {
            IGeometricMultigridModel coarserModel = CreateCoarserModel();
            (DokRowMajor restrictionMatrix, DokRowMajor interpolationMatrix) = CreateRestrictionAndInterpolationMatrix(coarserModel);
            return (coarserModel, restrictionMatrix, interpolationMatrix);
        }

        public (DokRowMajor A, Vector b) CreateLinearSystem()
        {
            throw new NotImplementedException(); //TODO: it MUST be implemented
        }

        public (DokRowMajor restrictionMatrix, DokRowMajor interpolationMatrix) CreateRestrictionAndInterpolationMatrix(IStructuredModel coarserModel)
        {
            throw new NotImplementedException(); //TODO: it MUST be implemented
        }

        public int[] FindFreeDofs()
        {
            throw new NotImplementedException(); //TODO: it MUST be implemented
        }

        public IStructuredModel GenerateModel(int detail)
        {
            throw new NotImplementedException(); //TODO: it MUST be implemented
        }

        public bool IsDofFree(int dof)
        {
            throw new NotImplementedException();
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
            NumDofsAll = Mesh.Dimension * Mesh.NumNodesTotal;
            NumDofsFree = NumDofsAll - NumDofsAll / Mesh.NumNodes[0]; // nodes with x = 0 are constrained
        }

        private static (Model model, UniformCartesianMesh2D mesh) CreateModel()
        {
            const int subdomainID = 0;
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
                new double[] { 0, 0 }, new double[] { Lx, Ly }, new int[] { numElemX, numElemY }).BuildMesh();

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




    }
}
