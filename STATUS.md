# Status

Current resident decode-stage policy: strict required CUDA.

Production MoE must route through the FlashInfer B12x AOT primitive. Rejected demo MoE paths have been removed from the production source and old benchmark/archeology artifacts have been deleted from the repo tree.

A clean runtime requires the AOT generated B12x backend/table/adapter archives. Missing required modules must fail package/link or initialization.
