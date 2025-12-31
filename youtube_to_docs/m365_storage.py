import asyncio
import io
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

import msal
import polars as pl
from kiota_abstractions.api_error import APIError
from msgraph import GraphServiceClient
from msgraph.generated.models.drive_item import DriveItem
from msgraph.generated.models.upload_session import UploadSession

from youtube_to_docs.storage import Storage


class AsyncRunner:
    """A helper class to run asyncio event loop in a background thread."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coro):
        """Run a coroutine in the background event loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def stop(self):
        """Stop the background event loop."""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()


class M365Storage(Storage):
    """Implementation of Storage for M365 (OneDrive)."""

    SCOPES = ["Files.ReadWrite.All", "User.Read"]

    def __init__(self, output_arg: str):
        self.runner = AsyncRunner()
        self.creds = self._get_creds()
        self.client = GraphServiceClient(credentials=self.creds, scopes=self.SCOPES)
        self.root_folder_id = self.runner.run(self._resolve_root_folder_id(output_arg))
        # Cache for folder IDs to avoid constant lookups
        self.folder_cache: dict[str, str] = {}
        # Cache for file metadata (path -> dict)
        self.file_cache: dict[str, DriveItem] = {}

    def __del__(self):
        self.runner.stop()

    def _get_creds(self):
        creds_file = Path.home() / ".ms_client_secret.json"

        if not creds_file.exists():
            raise FileNotFoundError(
                f"Client secrets not found at {creds_file.absolute()}"
            )

        with open(creds_file, "r") as f:
            client_config = json.load(f)

        app = msal.PublicClientApplication(
            client_id=client_config["client_id"],
            authority=client_config.get(
                "authority", "https://login.microsoftonline.com/common"
            ),
        )

        accounts = app.get_accounts()
        result = None

        if accounts:
            result = app.acquire_token_silent(self.SCOPES, account=accounts[0])

        if not result:
            flow = app.initiate_device_flow(scopes=self.SCOPES)
            if "user_code" not in flow:
                raise ValueError(
                    "Fail to create device flow. Err: %s" % json.dumps(flow, indent=4)
                )

            print(flow["message"])
            result = app.acquire_token_by_device_flow(flow)

        if "access_token" in result:
            return result

        raise Exception("Authentication failed", result)

    async def _resolve_root_folder_id(self, output_arg: str) -> str:
        if output_arg == "m365":
            folder_name = "youtube-to-docs-artifacts"
            try:
                root_folder = await self.client.me.drive.root.get()
                drive_items = await self.client.me.drive.root.children.get()
                if drive_items and drive_items.value:
                    for item in drive_items.value:
                        if item.name == folder_name:
                            print(f"Using existing folder: {folder_name} ({item.id})")
                            return item.id
                # Create folder
                new_folder = DriveItem(name=folder_name, folder={})
                created_folder = await self.client.me.drive.items[
                    root_folder.id
                ].children.post(new_folder)
                print(f"Created folder: {folder_name} ({created_folder.id})")
                return created_folder.id
            except APIError as e:
                print(f"Error resolving root folder: {e}")
                raise
        else:
            return output_arg

    async def _get_parent_id(self, path: str, create_path: bool = False) -> str:
        parts = Path(path).parts
        parent_id = self.root_folder_id

        for part in parts[:-1] if not create_path else parts:
            if part == ".":
                continue

            cache_key = f"{parent_id}/{part}"
            if cache_key in self.folder_cache:
                parent_id = self.folder_cache[cache_key]
                continue

            try:
                children = await self.client.me.drive.items[parent_id].children.get()
                found = False
                if children and children.value:
                    for item in children.value:
                        if item.name == part:
                            parent_id = item.id
                            self.folder_cache[cache_key] = parent_id
                            found = True
                            break
                if not found:
                    new_folder = DriveItem(name=part, folder={})
                    created_folder = await self.client.me.drive.items[
                        parent_id
                    ].children.post(new_folder)
                    parent_id = created_folder.id
                    self.folder_cache[cache_key] = parent_id
            except APIError as e:
                print(f"Error getting/creating parent folder: {e}")
                raise
        return parent_id

    async def _get_file_metadata(self, path: str) -> Optional[DriveItem]:
        if path in self.file_cache:
            return self.file_cache[path]

        parent_id = await self._get_parent_id(path)
        filename = Path(path).name

        try:
            children = await self.client.me.drive.items[parent_id].children.get()
            if children and children.value:
                for item in children.value:
                    if item.name == filename:
                        self.file_cache[path] = item
                        return item
        except APIError as e:
            if e.response.status_code != 404:
                print(f"Error getting file metadata: {e}")
            return None
        return None

    async def _get_file_id(self, path: str) -> Optional[str]:
        metadata = await self._get_file_metadata(path)
        return metadata.id if metadata else None

    def exists(self, path: str) -> bool:
        return self.runner.run(self._get_file_id(path)) is not None

    def read_text(self, path: str) -> str:
        return self.read_bytes(path).decode("utf-8")

    def read_bytes(self, path: str) -> bytes:
        return self.runner.run(self._read_bytes_async(path))

    async def _read_bytes_async(self, path: str) -> bytes:
        file_id = await self._get_file_id(path)
        if not file_id:
            raise FileNotFoundError(f"File not found: {path}")

        try:
            content = await self.client.me.drive.items[file_id].content.get()
            return content
        except APIError as e:
            print(f"Error reading file content: {e}")
            raise

    def write_text(self, path: str, content: str) -> str:
        return self.write_bytes(path, content.encode("utf-8"))

    def write_bytes(self, path: str, content: bytes) -> str:
        return self.runner.run(self._write_bytes_async(path, content))

    async def _write_bytes_async(self, path: str, content: bytes) -> str:
        parent_id = await self._get_parent_id(path)
        filename = Path(path).name

        try:
            item = await self.client.me.drive.items[parent_id].children[
                filename
            ].content.put(io.BytesIO(content))
            self.file_cache[path] = item
            return item.web_url
        except APIError as e:
            print(f"Error writing file: {e}")
            raise

    def load_dataframe(self, path: str) -> Optional[pl.DataFrame]:
        try:
            content = self.read_bytes(path)
            return pl.read_csv(io.BytesIO(content))
        except (FileNotFoundError, Exception) as e:
            print(f"Error loading dataframe: {e}")
            return None

    def save_dataframe(self, df: pl.DataFrame, path: str) -> str:
        csv_buffer = io.BytesIO()
        df.write_csv(csv_buffer)
        csv_buffer.seek(0)
        return self.write_bytes(path, csv_buffer.getvalue())

    def ensure_directory(self, path: str) -> None:
        self.runner.run(self._get_parent_id(path, create_path=True))

    def upload_file(
        self, local_path: str, target_path: str, content_type: Optional[str] = None
    ) -> str:
        with open(local_path, "rb") as f:
            content = f.read()
        return self.write_bytes(target_path, content)

    def get_full_path(self, path: str) -> str:
        metadata = self.runner.run(self._get_file_metadata(path))
        return metadata.web_url if metadata and metadata.web_url else path

    def get_local_file(
        self, path: str, download_dir: Optional[str] = None
    ) -> Optional[str]:
        if not self.exists(path):
            return None

        if not download_dir:
            import tempfile

            download_dir = tempfile.gettempdir()

        filename = Path(path).name
        local_path = os.path.join(download_dir, filename)

        try:
            data = self.read_bytes(path)
            with open(local_path, "wb") as f:
                f.write(data)
            return local_path
        except (APIError, OSError) as e:
            print(f"Error downloading file {path} to local: {e}")
            return None
