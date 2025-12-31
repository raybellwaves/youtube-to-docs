import asyncio
import io
import json
import os
import re
import shutil
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import msal
import polars as pl
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload
from kiota_abstractions.api_error import APIError
from msgraph import GraphServiceClient
from msgraph.generated.models.drive_item import DriveItem
from msgraph.generated.models.upload_session import UploadSession


class Storage(ABC):
    """Abstract base class for file storage operations."""

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Checks if a file exists at the given path."""
        pass

    @abstractmethod
    def read_text(self, path: str) -> str:
        """Reads text content from a file."""
        pass

    @abstractmethod
    def read_bytes(self, path: str) -> bytes:
        """Reads byte content from a file."""
        pass

    @abstractmethod
    def write_text(self, path: str, content: str) -> str:
        """Writes text content to a file. Returns the path or link to the file."""
        pass

    @abstractmethod
    def write_bytes(self, path: str, content: bytes) -> str:
        """Writes byte content to a file. Returns the path or link to the file."""
        pass

    @abstractmethod
    def load_dataframe(self, path: str) -> Optional[pl.DataFrame]:
        """Loads a DataFrame from a CSV file (or Sheet)."""
        pass

    @abstractmethod
    def save_dataframe(self, df: pl.DataFrame, path: str) -> str:
        """Saves a DataFrame to a CSV file (or Sheet). Returns the path or link."""
        pass

    @abstractmethod
    def ensure_directory(self, path: str) -> None:
        """Ensures a directory exists."""
        pass

    @abstractmethod
    def upload_file(
        self, local_path: str, target_path: str, content_type: Optional[str] = None
    ) -> str:
        """Uploads a local file to the storage. Returns the path or link."""
        pass

    @abstractmethod
    def get_full_path(self, path: str) -> str:
        """Returns the full path or link to the file."""
        pass

    @abstractmethod
    def get_local_file(
        self, path: str, download_dir: Optional[str] = None
    ) -> Optional[str]:
        """
        Ensures the file is available locally.
        If it's already local, returns the path.
        If it's remote, downloads it to download_dir (or a temp file) and returns
        the path.
        Returns None if retrieval fails.
        """
        pass


class LocalStorage(Storage):
    """Implementation of Storage for the local filesystem."""

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def read_text(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def read_bytes(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def write_text(self, path: str, content: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return os.path.abspath(path)

    def write_bytes(self, path: str, content: bytes) -> str:
        with open(path, "wb") as f:
            f.write(content)
        return os.path.abspath(path)

    def load_dataframe(self, path: str) -> Optional[pl.DataFrame]:
        try:
            return pl.read_csv(path)
        except Exception:
            return None

    def save_dataframe(self, df: pl.DataFrame, path: str) -> str:
        df.write_csv(path)
        return os.path.abspath(path)

    def ensure_directory(self, path: str) -> None:
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def upload_file(
        self, local_path: str, target_path: str, content_type: Optional[str] = None
    ) -> str:
        if os.path.abspath(local_path) != os.path.abspath(target_path):
            shutil.copy2(local_path, target_path)
        return os.path.abspath(target_path)

    def get_full_path(self, path: str) -> str:
        return os.path.abspath(path)

    def get_local_file(
        self, path: str, download_dir: Optional[str] = None
    ) -> Optional[str]:
        if self.exists(path):
            return os.path.abspath(path)
        return None


class GoogleDriveStorage(Storage):
    """Implementation of Storage for Google Drive."""

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    def __init__(self, output_arg: str):
        self.creds = self._get_creds()
        self.service = build("drive", "v3", credentials=self.creds)
        self.sheets_service = build("sheets", "v4", credentials=self.creds)
        self.root_folder_id = self._resolve_root_folder_id(output_arg)
        # Cache for folder IDs to avoid constant lookups
        self.folder_cache: dict[str, str] = {}
        # Cache for file metadata (path -> dict)
        self.file_cache: dict[str, dict] = {}

    def _get_creds(self):
        creds = None
        creds_file = Path.home() / ".google_client_secret.json"
        token_file = Path.home() / ".token.json"

        if token_file.exists():
            creds = Credentials.from_authorized_user_file(token_file, self.SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not creds_file.exists():
                    raise FileNotFoundError(
                        f"Client secrets not found at {creds_file.absolute()}"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(creds_file), self.SCOPES
                )
                creds = flow.run_local_server(port=0)
            token_file.write_text(creds.to_json())
        return creds

    def _resolve_root_folder_id(self, output_arg: str) -> str:
        if output_arg == "workspace":
            folder_name = "youtube-to-docs-artifacts"
            # specific logic to find or create "youtube-to-docs-artifacts" in root
            query = (
                "mimeType='application/vnd.google-apps.folder' and "
                f"name='{folder_name}' and trashed=false"
            )
            results = (
                self.service.files().list(q=query, fields="files(id, name)").execute()
            )
            files = results.get("files", [])
            if files:
                print(f"Using existing folder: {folder_name} ({files[0]['id']})")
                return files[0]["id"]
            else:
                file_metadata = {
                    "name": folder_name,
                    "mimeType": "application/vnd.google-apps.folder",
                }
                folder = (
                    self.service.files()
                    .create(body=file_metadata, fields="id")
                    .execute()
                )
                print(f"Created folder: {folder_name} ({folder.get('id')})")
                return folder.get("id")
        else:
            # Assume it is a Folder ID
            return output_arg

    def _get_parent_id(self, path: str) -> str:
        """
        Given a 'path' which mimics os.path.join(base_dir, subfolder, filename),
        figure out the parent folder ID.
        We assume 'path' starts with the storage's 'virtual' paths.
        However, main.py passes paths joined with os.path.sep.
        We need to handle normalization.
        """
        if path.startswith("http"):
            raise ValueError(f"Cannot resolve parent folder for URL: {path}")

        parts = Path(path).parts
        parent_id = self.root_folder_id

        # Iterate over parts except the last one (filename)
        for part in parts[:-1]:
            if part == "." or part == "youtube-to-docs-artifacts":  # Root or redundant
                continue

            cache_key = f"{parent_id}/{part}"
            if cache_key in self.folder_cache:
                parent_id = self.folder_cache[cache_key]
                continue

            # Search for folder in parent_id
            query = (
                "mimeType='application/vnd.google-apps.folder' and "
                f"name='{part}' and '{parent_id}' in parents and trashed=false"
            )
            results = self.service.files().list(q=query, fields="files(id)").execute()
            files = results.get("files", [])

            if files:
                current_id = files[0]["id"]
            else:
                # Create it
                file_metadata = {
                    "name": part,
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": [parent_id],
                }
                folder = (
                    self.service.files()
                    .create(body=file_metadata, fields="id")
                    .execute()
                )
                current_id = folder.get("id")

            self.folder_cache[cache_key] = current_id
            parent_id = current_id

        return parent_id

    def _get_file_metadata(self, path: str) -> Optional[dict]:
        if path in self.file_cache:
            return self.file_cache[path]

        # See if we can find the file given the path structure
        # path is like "summary-files/my-video.txt"
        parent_id = self._get_parent_id(path)
        filename = Path(path).name

        # Handle "Google Sheet" name special case for main csv
        # If path ends in .csv, we might look for a Sheet
        if filename == "youtube-docs.csv":
            filename = "youtube-docs"  # We store it as a Sheet

        query = f"name='{filename}' and '{parent_id}' in parents and trashed=false"
        results = (
            self.service.files()
            .list(q=query, fields="files(id, webViewLink, mimeType)")
            .execute()
        )
        files = results.get("files", [])
        if files:
            self.file_cache[path] = files[0]
            return files[0]
        return None

    def _get_file_id(self, path: str) -> Optional[str]:
        metadata = self._get_file_metadata(path)
        return metadata["id"] if metadata else None

    def _extract_id_from_url(self, url: str) -> Optional[str]:
        # Typical patterns:
        # https://docs.google.com/document/d/<ID>/edit
        # https://docs.google.com/spreadsheets/d/<ID>/edit
        # https://drive.google.com/file/d/<ID>/view
        match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
        if match:
            return match.group(1)
        return None

    def exists(self, path: str) -> bool:
        # Check if path starts with http - it's a link we already stored
        if path.startswith("http"):
            return True  # Assume links exist
        return self._get_file_id(path) is not None

    def read_text(self, path: str) -> str:
        # This implies downloading the Google Doc as text/markdown
        if path.startswith("http"):
            file_id = self._extract_id_from_url(path)
        else:
            file_id = self._get_file_id(path)

        if not file_id:
            raise FileNotFoundError(f"File not found: {path} (extracted id: {file_id})")

        # Export as plain text
        # If it is a google doc
        # mimeType = application/vnd.google-apps.document
        # export mimeType = text/plain
        try:
            content = (
                self.service.files()
                .export(fileId=file_id, mimeType="text/plain")
                .execute()
            )
            return content.decode("utf-8")
        except HttpError:
            # Maybe it is a binary file or text file? use get_media
            # But we are in read_text
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()

            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            return fh.getvalue().decode("utf-8")

    def read_bytes(self, path: str) -> bytes:
        if path.startswith("http"):
            file_id = self._extract_id_from_url(path)
        else:
            file_id = self._get_file_id(path)

        if not file_id:
            raise FileNotFoundError(f"File not found: {path}")

        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()

        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        return fh.getvalue()

    def write_text(self, path: str, content: str) -> str:
        parent_id = self._get_parent_id(path)
        filename = Path(path).name

        # Check if exists to overwrite?
        existing_id = self._get_file_id(path)

        file_metadata = {
            "name": filename,
            "mimeType": "application/vnd.google-apps.document",  # Convert to Doc
            "parents": [parent_id],
        }

        fh = io.BytesIO(content.encode("utf-8"))
        media = MediaIoBaseUpload(fh, mimetype="text/markdown", resumable=True)

        if existing_id:
            # Update
            file_metadata.pop("parents", None)  # Can't reparent easily on update
            file = (
                self.service.files()
                .update(
                    fileId=existing_id,
                    body=file_metadata,
                    media_body=media,
                    fields="id, webViewLink",
                )
                .execute()
            )
        else:
            file = (
                self.service.files()
                .create(body=file_metadata, media_body=media, fields="id, webViewLink")
                .execute()
            )

        # Update cache
        if file.get("id"):
            self.file_cache[path] = file

        return file.get("webViewLink")

    def write_bytes(self, path: str, content: bytes) -> str:
        parent_id = self._get_parent_id(path)
        filename = Path(path).name

        existing_id = self._get_file_id(path)

        file_metadata = {
            "name": filename,
            "parents": [parent_id],
        }

        fh = io.BytesIO(content)
        mime_type = "application/octet-stream"
        if filename.endswith(".wav"):
            mime_type = "audio/wav"
        elif filename.endswith(".png"):
            mime_type = "image/png"
        media = MediaIoBaseUpload(fh, mimetype=mime_type, resumable=True)

        if existing_id:
            file_metadata.pop("parents", None)
            file = (
                self.service.files()
                .update(
                    fileId=existing_id,
                    body=file_metadata,
                    media_body=media,
                    fields="id, webViewLink",
                )
                .execute()
            )
        else:
            file = (
                self.service.files()
                .create(body=file_metadata, media_body=media, fields="id, webViewLink")
                .execute()
            )

        # Update cache
        if file.get("id"):
            self.file_cache[path] = file

        return file.get("webViewLink")

    def load_dataframe(self, path: str) -> Optional[pl.DataFrame]:
        # path is expected to be "youtube-docs.csv" usually
        # We look for "youtube-docs" Sheet
        file_id = self._get_file_id(path)
        if not file_id:
            return None

        # Export sheet to CSV
        try:
            csv_content = (
                self.service.files()
                .export(fileId=file_id, mimeType="text/csv")
                .execute()
            )
            return pl.read_csv(io.BytesIO(csv_content))
        except Exception as e:
            print(f"Error loading dataframe from drive: {e}")
            return None

    def save_dataframe(self, df: pl.DataFrame, path: str) -> str:
        parent_id = self._get_parent_id(path)
        filename = Path(path).stem  # youtube-docs

        # Convert DF to CSV string
        csv_buffer = io.BytesIO()
        df.write_csv(csv_buffer)
        csv_buffer.seek(0)

        existing_id = self._get_file_id(path)

        file_metadata = {
            "name": filename,
            "mimeType": "application/vnd.google-apps.spreadsheet",
            "parents": [parent_id],
        }

        media = MediaIoBaseUpload(csv_buffer, mimetype="text/csv", resumable=True)

        if existing_id:
            # Update
            file_metadata.pop("parents", None)
            file = (
                self.service.files()
                .update(
                    fileId=existing_id,
                    body=file_metadata,
                    media_body=media,
                    fields="id, webViewLink",
                )
                .execute()
            )
        else:
            file = (
                self.service.files()
                .create(
                    body=file_metadata,
                    media_body=media,
                    fields="id, webViewLink",
                )
                .execute()
            )

        # Update sheet properties (freeze header and ensure min rows)
        try:
            spreadsheet_id = file.get("id")
            # Fetch the spreadsheet to get the actual sheetId of the first sheet
            spreadsheet = (
                self.sheets_service.spreadsheets()
                .get(spreadsheetId=spreadsheet_id)
                .execute()
            )
            sheets = spreadsheet.get("sheets", [])
            if sheets:
                sheet = sheets[0]
                sheet_id = sheet.get("properties", {}).get("sheetId")
                grid_props = sheet.get("properties", {}).get("gridProperties", {})
                current_rows = grid_props.get("rowCount", 0)

                if sheet_id is not None:
                    requests = []
                    # Freeze rows/cols
                    requests.append(
                        {
                            "updateSheetProperties": {
                                "properties": {
                                    "sheetId": sheet_id,
                                    "gridProperties": {
                                        "frozenRowCount": 1,
                                        "frozenColumnCount": 2,
                                    },
                                },
                                "fields": (
                                    "gridProperties(frozenRowCount,frozenColumnCount)"
                                ),
                            }
                        }
                    )

                    # Ensure minimum rows (1000) for better UX
                    if current_rows < 1000:
                        requests.append(
                            {
                                "updateSheetProperties": {
                                    "properties": {
                                        "sheetId": sheet_id,
                                        "gridProperties": {
                                            "rowCount": 1000,
                                        },
                                    },
                                    "fields": "gridProperties(rowCount)",
                                }
                            }
                        )

                    self.sheets_service.spreadsheets().batchUpdate(
                        spreadsheetId=spreadsheet_id, body={"requests": requests}
                    ).execute()
        except Exception as e:
            print(f"Warning: Could not update sheet properties: {e}")

        # Update cache
        if file.get("id"):
            self.file_cache[path] = file

        return file.get("webViewLink")

    def ensure_directory(self, path: str) -> None:
        # _get_parent_id creates directories as side effect
        self._get_parent_id(os.path.join(path, "dummy"))

    def upload_file(
        self, local_path: str, target_path: str, content_type: Optional[str] = None
    ) -> str:
        parent_id = self._get_parent_id(target_path)
        filename = Path(target_path).name

        existing_id = self._get_file_id(target_path)

        file_metadata = {"name": filename, "parents": [parent_id]}

        media = MediaFileUpload(local_path, mimetype=content_type, resumable=True)

        if existing_id:
            file_metadata.pop("parents", None)
            file = (
                self.service.files()
                .update(
                    fileId=existing_id,
                    body=file_metadata,
                    media_body=media,
                    fields="id, webViewLink",
                )
                .execute()
            )
        else:
            file = (
                self.service.files()
                .create(body=file_metadata, media_body=media, fields="id, webViewLink")
                .execute()
            )

        # Update cache
        if file.get("id"):
            self.file_cache[target_path] = file

        return file.get("webViewLink")

    def get_full_path(self, path: str) -> str:
        """For Drive, returns the path (or link if possible/cached)."""
        if path.startswith("http"):
            return path

        metadata = self._get_file_metadata(path)
        if metadata and "webViewLink" in metadata:
            return metadata["webViewLink"]
        return path

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
        except (HttpError, OSError) as e:
            print(f"Error downloading file {path} to local: {e}")
            return None


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
