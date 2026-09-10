"""Identity and geometry profiles for the DSV4 GA stage-0 reference fixture.

Each profile pins one checkpoint (model id, revision, identity-file and
source-shard SHA-256 set) plus the fixed tensor geometry the retained
fixture is verified against. The flash values were carried over verbatim
from the pre-profile constants of the generator and verifier; the pro
values were measured by tools/dsv4pro_ga_profile_identity.sh
(queue receipt dsv4pro-ga-profile-identity, spark6, exit 0).
"""

PROFILES = {
	"flash": {
		"model": "deepseek-ai/DeepSeek-V4-Flash-0731",
		"revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
		"index_sha256": "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
		"config_sha256": "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
		"tokenizer_sha256": "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
		"reference_model_sha256": "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f",
		"reference_kernel_sha256": "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2",
		"reference_config_sha256": "c90861f3d10a9e4ef5954f8f1a34c529d480da1c5799f84660028f4e38e14e71",
		"batch_json_sha256": "6f7836819a9ecdbca117b18cb4717aa8cb91c230af5961c5d025968cef34f8bb",
		"token_payload_sha256": "f2f860f7843e755c4cdfcea408c647559ab604fde5c34a00bac314ba62289769",
		"token_count": 128,
		"prompt_tokens": 128,
		"first_layer": 0,
		"layer_count": 3,
		"hc_stream_count": 4,
		"hidden_dimension": 4096,
		"vocabulary_size": 129280,
		"source_shards": {
			"model-00001-of-00048.safetensors": (1059061856, "f3668ba4cccf1ca6a7eb84e888fb92c1cdc7204d472ba9db771e6fd3abf6b874"),
			"model-00002-of-00048.safetensors": (3566321192, "77b26c939a0e25b3113c8d6bb04e1901a748bd4a7d2589e3bfdaabdf1e9bba14"),
			"model-00003-of-00048.safetensors": (3566321192, "412abf4c906faadc221ef0cb50f90fe20bde8454a08ad4dc2364b6b79e7fda5c"),
			"model-00004-of-00048.safetensors": (3596229272, "9610f56bc587fb0ff9a8b68a60299482ee8c433fe5b5587e4257aca98add4a2e"),
		},
	},
	"pro": {
		"model": "deepseek-ai/DeepSeek-V4-Pro-0813 (HF, GA 2026-08-13)",
		"revision": "2de2ac1e43134f8b03bf6156067715b7c3c73b1a507329e606023c601a56d30a",
		"index_sha256": "2de2ac1e43134f8b03bf6156067715b7c3c73b1a507329e606023c601a56d30a",
		"config_sha256": "9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0",
		"tokenizer_sha256": "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
		"reference_model_sha256": "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f",
		"reference_kernel_sha256": "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2",
		"reference_config_sha256": "801bc719d08cd5be57cddc185cac621417522810e29d0314849c939bf0176ee7",
		"batch_json_sha256": "3991622bba754df3a39358c3261a20d7acb08a4cc23b21549fdd4e7f85c1fe9f",
		"token_payload_sha256": "PENDING-PRO-GENERATION",
		"token_count": 128,
		"prompt_tokens": 128,
		"first_layer": 0,
		"layer_count": 3,
		"hc_stream_count": 4,
		"hidden_dimension": 7168,
		"vocabulary_size": 129280,
		"source_shards": {
			"model-00001-of-00066.safetensors": (1853358176, "74ff60e304294ef4482341d941dafcd0143e49fd3d4569f283b1e9020301c947"),
			"model-00002-of-00066.safetensors": (13874726024, "d7be184a5582a5a5a7b430c92572996ed8e43f2fe340f96b63a8dcd74ca3e52a"),
			"model-00003-of-00066.safetensors": (13874726024, "fc9291c02445cdb028315258f3bd9a65e336ebcfc45b74a5e0c78907b9ad7b47"),
			"model-00004-of-00066.safetensors": (13910006752, "6f2c6511e5cc9d611be87963ab95519ab38a565abba48c813dccac2208bc655b"),
		},
	},
}


def profile(name):
	if name not in PROFILES:
		raise KeyError(name)
	return PROFILES[name]
