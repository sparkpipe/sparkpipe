import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('compile_tokenizer',ROOT/'tools/spark_tiktoken_compile.py')
compiler=importlib.util.module_from_spec(spec);spec.loader.exec_module(compiler)


class CompileTests(unittest.TestCase):
    def test_real_loader_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pieces=[bytes([b]) for b in range(256)]+[b'He',b'll',b'Hell',b'Hello',b'12',b'123',b'23',b'1234',b'wor',b'wo',b'world',b'rl',b'ld']
            (root/'ranks').write_bytes(b''.join(base64.b64encode(piece)+b' '+str(i).encode()+b'\n' for i,piece in enumerate(pieces)))
            (root/'config').write_text(json.dumps(dict(added_tokens_decoder={'400':dict(content='[EOS]',special=True),'399':dict(content='[OPEN]',special=False)})))
            result=compiler.compile_tokenizer(root/'ranks',root/'config',root/'compiled')
            self.assertEqual(result['vocabulary_size'],401)
            subprocess.run([str(ROOT/'build/test_tiktoken_compiled'),str(root/'ranks'),str(root/'compiled')],check=True)
            with self.assertRaises(FileExistsError):compiler.compile_tokenizer(root/'ranks',root/'config',root/'compiled')

    def test_invalid_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'ranks').write_text('YQ== 1\n');(root/'config').write_text('{}')
            with self.assertRaisesRegex(ValueError,'contiguous'):compiler.compile_tokenizer(root/'ranks',root/'config',root/'compiled')
            self.assertFalse((root/'compiled').exists())


if __name__=='__main__':unittest.main()
