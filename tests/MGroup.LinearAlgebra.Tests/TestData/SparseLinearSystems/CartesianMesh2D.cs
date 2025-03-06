using MGroup.MSolve.Discretization;

namespace Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems
{
    using System;

    public class CartesianMesh2D : CartesianMeshBase
    {
        public CartesianMesh2D(int[] numElementsPerAxis, double[] lengthPerAxis) : base(numElementsPerAxis, lengthPerAxis) { }

        public override int[] GetNodeIdsOfElement(int[] elementIdx)
        {
            return new int[]
            {
                GetNodeID(new int[] { elementIdx[0], elementIdx[1] }),
                GetNodeID(new int[] { elementIdx[0] + 1, elementIdx[1] }),
                GetNodeID(new int[] { elementIdx[0] + 1, elementIdx[1] + 1 }),
                GetNodeID(new int[] { elementIdx[0], elementIdx[1] + 1 })
            };
        }

        public override int GetElementID(int[] elementIdx) => numElementsPerAxis[1] * elementIdx[0] + elementIdx[1];

        public override int[] GetElementIdx(int elementID) => new int[] { elementID / numElementsPerAxis[1], elementID % numElementsPerAxis[1] };

        public override int GetNodeID(int[] nodeIdx) => numNodesPerAxis[1] * nodeIdx[0] + nodeIdx[1];

        public override int[] GetNodeIdx(int nodeID) => new int[] { nodeID / numNodesPerAxis[1], nodeID % numNodesPerAxis[1] };

        public override CellType CellType => CellType.Quad4;
        public override int NumNodesPerElement => 4;

        public override int[] GetElementConnectivity(int[] elementIdx) => throw new NotImplementedException();
        public override int[] GetElementConnectivity(int elementID) => throw new NotImplementedException();
        public override double[] MinCoordinates => throw new NotImplementedException();
        public override double[] MaxCoordinates => throw new NotImplementedException();
        public override int[] NumNodes => throw new NotImplementedException();
    }
}
