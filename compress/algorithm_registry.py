from algorithms import (
    AerCompressor,
    DeltaRleCompressor,
    DeltaSparseCompressor,
    DeltaSparseZlibCompressor,
    RleCompressor,
    TemporalBinningCompressor,
)


def build_compressor(algorithm_id, params=None):
    normalized_id = str(algorithm_id).strip().lower()
    params = dict(params or {})

    if normalized_id == "rle":
        return RleCompressor()
    if normalized_id == "delta_rle":
        return DeltaRleCompressor()
    if normalized_id == "delta_sparse":
        return DeltaSparseCompressor()
    if normalized_id == "delta_sparse_zlib":
        return DeltaSparseZlibCompressor()
    if normalized_id == "aer":
        return AerCompressor(use_delta=bool(params.get("use_delta", False)))
    if normalized_id == "temporal_binning":
        return TemporalBinningCompressor(bin_size=int(params.get("bin_size", 255)))

    raise ValueError(f"Unsupported algorithm id: {algorithm_id}")


def build_output_filename(compressor):
    return "compressed.bin"
