using MGroup.LinearAlgebra.Matrices.Builders;

namespace Compression.src.MGroup.Solvers.Multigrid
{
    public interface IGeometricMultigridModel : IStructuredModel
    {
        //TODO: μελλοντικά, είναι δυνατόν, το model, να δίνει για κάθε coarser node ή DoF (array) ένα array με τους finerNodes ή finerDoFs που τον
        // επηρρεάζουν καθώς και πόσο τον επηρρεάζουν. Αυτό θα περιλαμβάνει τόσο για τους coarser όσο για για τους finer και nodes ή DoFs που θα είναι
        // παγιωμένοι. Με αυτό το array σε array, είναι πλέον εύκολο να φτιαχτούν οι restriction/interpolation πίνακες για κάθε είδους μοντέλο σε
        // επίπεδο αλγόριθμου και όχι σε επίπεδο μοντέλου. Τι γλιτώνουμε; Όχι πάρα πολλά γιατί αν έχουμε αυτό το δεδομένο, τα υπόλοιπα που μένουν
        // για να ολοκληρωθούν οι restriction/interpolation πίνακες είναι να γίνουν γραμμές/στήλες normalized στο 100% = 1 και στη συνέχεια να
        // αφαιρεθούν οι γραμμές που αφορούν fixed DoFs.

        /// <summary>
        /// It generates the restriction and the interpolation matrix for current model.
        /// </summary>
        /// <param name="coarserModel">The immediately coarser model after current model. Even it can be generated from the model, to avoid the
        /// overhead of generation, it is taken as parameter (in case which already is generated). Otherwise use currentModel.GenerateModel(-1) or
        /// currentModel.CreateCoarserModel()</param>
        /// <returns>The first element of tuple is the restriction matrix which goes from current model to the coarser one.
        /// The second element of tuple is the interpolation matrix which goes from the coarser model to current model (not from the current model to
        /// finer one).</returns>
        public (DokRowMajor restrictionMatrix, DokRowMajor interpolationMatrix) CreateRestrictionAndInterpolationMatrix(IStructuredModel coarserModel);

        /// <summary>
        /// It creates a coarser model with half the length on each dimension.
        /// </summary>
        /// <remarks>It is the same with <see cref="IStructuredModel.GenerateModel(int)"/></remarks>
        /// <returns>The coarser model.</returns>
        public IGeometricMultigridModel CreateCoarserModel();

        /// <summary>
        /// It creates a coarser model with half the length on each dimension and corresponding restriction and interpolation matrix.
        /// </summary>
        /// <remarks>It is the same with <see cref="CreateCoarserModel()"/> followed by <see cref="RestrictionAndInterpolationMatrix()"/></remarks>
        /// <returns>First element is the coarser model.
        /// Second element is the restriction matrix from current model to coarser model.
        /// Third element is the interpolation matrix from coarser model to current model.</returns>
        public (IGeometricMultigridModel coarserModel, DokRowMajor restrictionMatrix, DokRowMajor interpolationMatrix) CreateCoarserModelAndSmoothenerMatrices();
    }
}
