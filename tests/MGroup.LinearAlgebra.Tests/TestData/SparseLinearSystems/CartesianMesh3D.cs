using MGroup.MSolve.Discretization;

namespace Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems
{
    using System;

    public class CartesianMesh3D : CartesianMeshBase
    {
        public CartesianMesh3D(int[] numElementsPerAxis, double[] lengthPerAxis) : base(numElementsPerAxis, lengthPerAxis) {}

      


        public override int[] GetNodeIdsOfElement(int[] elementIdx)
        {
            return new int[]
            {
                GetNodeID(new int[] { elementIdx[0], elementIdx[1], elementIdx[2] }),
                GetNodeID(new int[] { elementIdx[0] + 1, elementIdx[1], elementIdx[2] }),
                GetNodeID(new int[] { elementIdx[0] + 1, elementIdx[1] + 1, elementIdx[2] }),
                GetNodeID(new int[] { elementIdx[0], elementIdx[1] + 1, elementIdx[2] }),
                GetNodeID(new int[] { elementIdx[0], elementIdx[1], elementIdx[2] + 1 }),
                GetNodeID(new int[] { elementIdx[0] + 1, elementIdx[1], elementIdx[2] + 1 }),
                GetNodeID(new int[] { elementIdx[0] + 1, elementIdx[1] + 1, elementIdx[2] + 1 }),
                GetNodeID(new int[] { elementIdx[0], elementIdx[1] + 1, elementIdx[2] + 1 })
            };
        }

        public override int GetElementID(int[] elementIdx)
            => elementIdx[1] + elementIdx[2] * numElementsPerAxis[1] + elementIdx[0] * numElementsPerAxis[1] * numElementsPerAxis[2];

        public override int[] GetElementIdx(int elementID)
        {
            int[] idx = new int[3];
            int numElementsPlane = numElementsPerAxis[1] * numElementsPerAxis[2];
            int mod = elementID % numElementsPlane;
            idx[0] = elementID / numElementsPlane;
            idx[1] = mod % numElementsPerAxis[1];
            idx[2] = mod / numElementsPerAxis[1];
            return idx;
        }

        public override int GetNodeID(int[] nodeIdx) => nodeIdx[1] + nodeIdx[2] * numNodesPerAxis[1] + nodeIdx[0] * numNodesPerAxis[1] * numNodesPerAxis[2];

        public override int[] GetNodeIdx(int nodeID)
        {
            int numNodesPlane = numNodesPerAxis[1] * numNodesPerAxis[2];
            int mod = nodeID % numNodesPlane;
            int[] idx = new int[3];
            idx[0] = nodeID / numNodesPlane;
            idx[1] = mod % numNodesPerAxis[1];
            idx[2] = mod / numNodesPerAxis[1];
            return idx;
        }

        public override CellType CellType => CellType.Hexa8;
        public override int NumNodesPerElement => 8;

        public override int[] GetElementConnectivity(int[] elementIdx) => throw new NotImplementedException();
        public override int[] GetElementConnectivity(int elementID) => throw new NotImplementedException();
        public override double[] MinCoordinates => throw new NotImplementedException();
        public override double[] MaxCoordinates => throw new NotImplementedException();
        public override int[] NumNodes => throw new NotImplementedException();
    }
}
