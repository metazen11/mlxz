# Changelog

## 0.1.0 (2026-04-28)


### Features

* mlxz v1.0.0 — high-throughput MLX inference server for Apple Silicon ([943d181](https://github.com/metazen11/mlxz/commit/943d18130670b90eb54d7c2949ec3943b4460850))


### Performance Improvements

* adapt kv cache threshold to total request length ([#17](https://github.com/metazen11/mlxz/issues/17)) ([0741862](https://github.com/metazen11/mlxz/commit/0741862160055d39209f91d303cbe6de788fd3da))
* adaptive quantized KV policy with higher default threshold ([#19](https://github.com/metazen11/mlxz/issues/19)) ([24be607](https://github.com/metazen11/mlxz/commit/24be6077dd1094512453f457e1bab3d46faf0f5b))
* restore quantized cache offset for long prompts ([#18](https://github.com/metazen11/mlxz/issues/18)) ([94ef302](https://github.com/metazen11/mlxz/commit/94ef302ae26ca381425278f40a31c136726f39bc))
