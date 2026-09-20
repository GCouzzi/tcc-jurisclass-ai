import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.model_source import resolve_model_directory


class ModelSourceTests(unittest.TestCase):
    def test_local_directory_does_not_require_hub(self):
        self.assertEqual(resolve_model_directory("models"), Path("models"))

    def test_hub_requires_revision(self):
        with self.assertRaises(ValueError):
            resolve_model_directory("models", "user/model")

    def test_only_requested_fold_files_are_downloaded(self):
        download = Mock(return_value="cache/snapshot")
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            result = resolve_model_directory("models", "user/model", "abc123", [5])
        self.assertEqual(result, Path("cache/snapshot"))
        args = download.call_args.kwargs
        self.assertEqual(args["revision"], "abc123")
        self.assertTrue(all(p.startswith("fold_5/best_model/") for p in args["allow_patterns"]))
        self.assertFalse(any("training_args" in p for p in args["allow_patterns"]))
