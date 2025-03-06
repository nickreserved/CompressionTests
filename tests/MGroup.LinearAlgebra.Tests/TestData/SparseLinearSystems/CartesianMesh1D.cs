using MGroup.MSolve.Discretization;

namespace Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems
{
    using System;

    public class CartesianMesh1D : CartesianMeshBase
    {
        public CartesianMesh1D(int numElements, double length) : base(new int[] { numElements }, new double[] { length }) { }

        public CartesianMesh1D(int[] numElementsPerAxis, double[] lengthPerAxis) : base(numElementsPerAxis, lengthPerAxis) { }

        public override int[] GetNodeIdsOfElement(int[] elementIdx)
            => new int[]
            {
                GetNodeID(new int[] { elementIdx[0] }),
                GetNodeID(new int[] { elementIdx[0] + 1 })
            };

        public override int GetElementID(int[] elementIdx) => elementIdx[0];

        public override int[] GetElementIdx(int elementId) => new int[] { elementId };

        public override int GetNodeID(int[] nodeIdx) => nodeIdx[0];
       
        public override int[] GetNodeIdx(int nodeId) => new int[] { nodeId };


        public override CellType CellType => CellType.Line2;
        public override int NumNodesPerElement => 4;

        public override int[] GetElementConnectivity(int[] elementIdx) => throw new NotImplementedException();
        public override int[] GetElementConnectivity(int elementID) => throw new NotImplementedException();
        public override double[] MinCoordinates => throw new NotImplementedException();
        public override double[] MaxCoordinates => throw new NotImplementedException();
        public override int[] NumNodes => throw new NotImplementedException();
    }
}
