import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
from kiota_abstractions.api_error import APIError

from youtube_to_docs.m365_storage import M365Storage


class TestM365Storage(unittest.TestCase):
    @patch("youtube_to_docs.m365_storage.M365Storage._get_creds")
    @patch("youtube_to_docs.m365_storage.GraphServiceClient")
    def setUp(self, mock_graph_client, mock_get_creds):
        # Mock the credentials and GraphServiceClient
        mock_get_creds.return_value = MagicMock()
        self.mock_graph_client_instance = mock_graph_client.return_value

        # Mock the async resolve_root_folder_id to be called synchronously in the constructor
        with patch.object(
            M365Storage, "_resolve_root_folder_id", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = "root_folder_id"
            self.storage = M365Storage("m365")

    def test_initialization(self):
        self.assertIsNotNone(self.storage.client)
        self.assertEqual(self.storage.root_folder_id, "root_folder_id")

    def test_exists(self):
        with patch.object(self.storage, "_get_file_id", new_callable=AsyncMock) as mock_get_file_id:
            mock_get_file_id.return_value = "123"
            self.assertTrue(self.storage.exists("test.txt"))
            mock_get_file_id.return_value = None
            self.assertFalse(self.storage.exists("not_found.txt"))

    def test_read_text(self):
        with patch.object(self.storage, "read_bytes") as mock_read_bytes:
            mock_read_bytes.return_value = b"hello world"
            self.assertEqual(self.storage.read_text("test.txt"), "hello world")

    def test_read_bytes(self):
        with patch.object(self.storage, "_read_bytes_async", new_callable=AsyncMock) as mock_read_bytes_async:
            mock_read_bytes_async.return_value = b"hello world"
            self.assertEqual(self.storage.read_bytes("test.txt"), b"hello world")

    def test_write_text(self):
        with patch.object(self.storage, "write_bytes") as mock_write_bytes:
            mock_write_bytes.return_value = "http://purl.com/123"
            self.assertEqual(
                self.storage.write_text("test.txt", "hello world"), "http://purl.com/123"
            )

    def test_write_bytes(self):
        with patch.object(self.storage, "_write_bytes_async", new_callable=AsyncMock) as mock_write_bytes_async:
            mock_write_bytes_async.return_value = "http://purl.com/123"
            self.assertEqual(
                self.storage.write_bytes("test.txt", b"hello world"), "http://purl.com/123"
            )

    def test_load_dataframe(self):
        with patch.object(self.storage, "read_bytes") as mock_read_bytes:
            mock_read_bytes.return_value = b"col1,col2\n1,2\n3,4"
            df = self.storage.load_dataframe("test.csv")
            self.assertIsInstance(df, pl.DataFrame)
            self.assertEqual(df.shape, (2, 2))

    def test_save_dataframe(self):
        df = pl.DataFrame({"col1": [1, 3], "col2": [2, 4]})
        with patch.object(self.storage, "write_bytes") as mock_write_bytes:
            mock_write_bytes.return_value = "http://purl.com/123"
            self.assertEqual(self.storage.save_dataframe(df, "test.csv"), "http://purl.com/123")

    def test_ensure_directory(self):
        with patch.object(self.storage, "_get_parent_id", new_callable=AsyncMock) as mock_get_parent_id:
            self.storage.ensure_directory("test_dir")
            mock_get_parent_id.assert_called_once_with("test_dir", create_path=True)

    def test_upload_file(self):
        with patch("builtins.open", unittest.mock.mock_open(read_data=b"file content")):
            with patch.object(self.storage, "write_bytes") as mock_write_bytes:
                mock_write_bytes.return_value = "http://purl.com/123"
                self.assertEqual(
                    self.storage.upload_file("local.txt", "remote.txt"),
                    "http://purl.com/123",
                )

    def test_get_full_path(self):
        mock_metadata = MagicMock()
        mock_metadata.web_url = "http://purl.com/123"
        with patch.object(self.storage, "_get_file_metadata", new_callable=AsyncMock) as mock_get_file_metadata:
            mock_get_file_metadata.return_value = mock_metadata
            self.assertEqual(self.storage.get_full_path("test.txt"), "http://purl.com/123")

    def test_get_local_file(self):
        with patch.object(self.storage, "exists") as mock_exists, \
             patch.object(self.storage, "read_bytes") as mock_read_bytes, \
             patch("builtins.open", unittest.mock.mock_open()) as mock_file:

            mock_exists.return_value = True
            mock_read_bytes.return_value = b"file content"

            local_path = self.storage.get_local_file("test.txt")

            self.assertIsNotNone(local_path)
            mock_file.assert_called_with(local_path, "wb")
            mock_file().write.assert_called_once_with(b"file content")


if __name__ == "__main__":
    unittest.main()
