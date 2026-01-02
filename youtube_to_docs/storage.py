import atexit
import base64
import io
import json
import os
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from mimetypes import guess_type
from pathlib import Path
from typing import Optional

import msal
import polars as pl
import pypandoc
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload


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


class M365Storage(Storage):
    """Implementation of Storage for Microsoft 365 (OneDrive/SharePoint)."""

    GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
    SCOPES = ["Files.ReadWrite"]
    CLIENT_CONFIG_FILE = Path.home() / ".azure_client.json"
    TOKEN_CACHE_FILE = Path.home() / ".msal_token_cache.json"

    def __init__(self, output_arg: str):
        self.root_folder_name = (
            "youtube-to-docs-artifacts"
            if output_arg == "sharepoint"
            else output_arg
        )
        self.app = self._build_msal_app()
        self.root_folder_id = self._ensure_root_folder()
        self.folder_cache: dict[str, str] = {}
        self.file_cache: dict[str, dict] = {}

    def _build_msal_app(self) -> msal.PublicClientApplication:
        if not self.CLIENT_CONFIG_FILE.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {self.CLIENT_CONFIG_FILE}. "
                "Please create it with {'client_id': '...', 'authority': '...'}"
            )
        config = json.loads(self.CLIENT_CONFIG_FILE.read_text(encoding="utf-8"))
        client_id = config.get("client_id")
        authority = config.get("authority")
        if not client_id or not authority:
            raise ValueError(
                "Azure client config must include both 'client_id' and 'authority'."
            )

        cache = msal.SerializableTokenCache()
        if self.TOKEN_CACHE_FILE.exists():
            cache.deserialize(self.TOKEN_CACHE_FILE.read_text(encoding="utf-8"))

        def persist_cache() -> None:
            if cache.has_state_changed:
                self.TOKEN_CACHE_FILE.write_text(cache.serialize(), encoding="utf-8")

        atexit.register(persist_cache)

        return msal.PublicClientApplication(
            client_id=client_id,
            authority=authority,
            token_cache=cache,
        )

    def _get_access_token(self) -> str:
        accounts = self.app.get_accounts()
        result = None
        if accounts:
            result = self.app.acquire_token_silent(self.SCOPES, account=accounts[0])
        if not result:
            result = self.app.acquire_token_interactive(self.SCOPES)
        if "access_token" not in result:
            error_msg = (
                result.get("error_description") or result.get("error") or "Unknown error"
            )
            raise RuntimeError(f"Could not authenticate: {error_msg}")
        return result["access_token"]

    def _headers(self, content_type: Optional[str] = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._get_access_token()}"}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        resp = requests.request(method, url, **kwargs)
        if resp.status_code == 404:
            return resp
        resp.raise_for_status()
        return resp

    def _encode_share_url(self, url: str) -> str:
        encoded = base64.urlsafe_b64encode(url.encode("utf-8")).decode("utf-8")
        return f"u!{encoded.rstrip('=')}"

    def _normalize_path(self, path: str) -> str:
        parts = []
        for part in Path(path).parts:
            if part in (".", ""):
                continue
            if part == self.root_folder_name:
                continue
            parts.append(part)
        return "/".join(parts)

    def _map_text_path(self, path: str) -> str:
        if path.startswith("http"):
            return path
        suffix = Path(path).suffix.lower()
        if suffix in (".md", ".txt"):
            return str(Path(path).with_suffix(".docx"))
        return path

    def _map_dataframe_path(self, path: str) -> str:
        if path.startswith("http"):
            return path
        if path.lower().endswith(".csv"):
            return str(Path(path).with_suffix(".xlsx"))
        return path

    def _get_item_by_path(self, path: str) -> Optional[dict]:
        if path in self.file_cache:
            return self.file_cache[path]
        rel_path = self._normalize_path(path)
        if not rel_path:
            return None
        url = f"{self.GRAPH_BASE_URL}/me/drive/items/{self.root_folder_id}:/{rel_path}"
        resp = self._request("get", url, headers=self._headers())
        if resp.status_code == 404:
            return None
        data = resp.json()
        self.file_cache[path] = data
        return data

    def _get_item_by_url(self, url: str) -> Optional[dict]:
        share_id = self._encode_share_url(url)
        endpoint = f"{self.GRAPH_BASE_URL}/shares/{share_id}/driveItem"
        resp = self._request("get", endpoint, headers=self._headers())
        if resp.status_code == 404:
            return None
        return resp.json()

    def _ensure_root_folder(self) -> str:
        if not self.root_folder_name:
            raise ValueError("Root folder name cannot be empty.")

        url = f"{self.GRAPH_BASE_URL}/me/drive/root:/{self.root_folder_name}"
        resp = self._request("get", url, headers=self._headers())
        if resp.status_code == 404:
            create_url = f"{self.GRAPH_BASE_URL}/me/drive/root/children"
            payload = {
                "name": self.root_folder_name,
                "folder": {},
                "@microsoft.graph.conflictBehavior": "rename",
            }
            created = self._request(
                "post", create_url, headers=self._headers("application/json"), json=payload
            ).json()
            return created["id"]
        return resp.json()["id"]

    def _get_child_folder(self, parent_id: str, name: str) -> Optional[dict]:
        safe_name = name.replace("'", "''")
        url = (
            f"{self.GRAPH_BASE_URL}/me/drive/items/{parent_id}/children"
            f"?$filter=name eq '{safe_name}'"
        )
        resp = self._request("get", url, headers=self._headers())
        if resp.status_code == 404:
            return None
        items = resp.json().get("value", [])
        for item in items:
            if item.get("folder") is not None:
                return item
        return None

    def _create_folder(self, parent_id: str, name: str) -> dict:
        url = f"{self.GRAPH_BASE_URL}/me/drive/items/{parent_id}/children"
        payload = {
            "name": name,
            "folder": {},
            "@microsoft.graph.conflictBehavior": "rename",
        }
        return self._request(
            "post", url, headers=self._headers("application/json"), json=payload
        ).json()

    def _ensure_folder_path(self, path: str) -> str:
        normalized = self._normalize_path(path)
        if not normalized:
            return self.root_folder_id

        parent_id = self.root_folder_id
        current_path = ""
        for part in Path(normalized).parts:
            current_path = f"{current_path}/{part}" if current_path else part
            if current_path in self.folder_cache:
                parent_id = self.folder_cache[current_path]
                continue

            folder = self._get_child_folder(parent_id, part)
            if folder is None:
                folder = self._create_folder(parent_id, part)
            parent_id = folder["id"]
            self.folder_cache[current_path] = parent_id
        return parent_id

    def _upload_bytes(
        self, path: str, content: bytes, content_type: Optional[str] = None
    ) -> dict:
        rel_path = self._normalize_path(path)
        url = f"{self.GRAPH_BASE_URL}/me/drive/items/{self.root_folder_id}:/{rel_path}:/content"
        resp = self._request(
            "put",
            url,
            headers=self._headers(content_type),
            data=content,
        )
        return resp.json()

    def _download_bytes(self, item_id: str) -> bytes:
        url = f"{self.GRAPH_BASE_URL}/me/drive/items/{item_id}/content"
        resp = self._request("get", url, headers=self._headers())
        return resp.content

    def _convert_markdown_to_docx(self, markdown_text: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_docx:
            tmp_docx_path = tmp_docx.name
        try:
            pypandoc.convert_text(
                markdown_text,
                to="docx",
                format="md",
                outputfile=tmp_docx_path,
                extra_args=["--wrap=preserve"],
            )
            return Path(tmp_docx_path).read_bytes()
        finally:
            try:
                os.remove(tmp_docx_path)
            except FileNotFoundError:
                pass

    def _convert_docx_to_markdown(self, docx_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_docx:
            tmp_docx.write(docx_bytes)
            tmp_docx_path = tmp_docx.name
        try:
            return pypandoc.convert_file(tmp_docx_path, to="md", format="docx")
        finally:
            try:
                os.remove(tmp_docx_path)
            except FileNotFoundError:
                pass

    def exists(self, path: str) -> bool:
        if path.startswith("http"):
            return True
        mapped = self._map_dataframe_path(self._map_text_path(path))
        return self._get_item_by_path(mapped) is not None

    def read_text(self, path: str) -> str:
        if path.startswith("http"):
            item = self._get_item_by_url(path)
            if not item:
                raise FileNotFoundError(f"File not found: {path}")
        else:
            mapped_path = self._map_text_path(path)
            item = self._get_item_by_path(mapped_path)
            if not item:
                raise FileNotFoundError(f"File not found: {path}")

        file_name = item.get("name", "")
        content = self._download_bytes(item["id"])
        if file_name.lower().endswith(".docx"):
            return self._convert_docx_to_markdown(content)
        return content.decode("utf-8")

    def read_bytes(self, path: str) -> bytes:
        if path.startswith("http"):
            item = self._get_item_by_url(path)
            if not item:
                raise FileNotFoundError(f"File not found: {path}")
        else:
            mapped_path = self._map_dataframe_path(self._map_text_path(path))
            item = self._get_item_by_path(mapped_path)
            if not item:
                raise FileNotFoundError(f"File not found: {path}")
        return self._download_bytes(item["id"])

    def write_text(self, path: str, content: str) -> str:
        mapped_path = self._map_text_path(path)
        parent_dir = str(Path(mapped_path).parent)
        if parent_dir not in (".", ""):
            self._ensure_folder_path(parent_dir)

        docx_bytes = (
            self._convert_markdown_to_docx(content)
            if Path(mapped_path).suffix.lower() == ".docx"
            else content.encode("utf-8")
        )

        mime_type = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            if Path(mapped_path).suffix.lower() == ".docx"
            else "text/plain"
        )
        item = self._upload_bytes(mapped_path, docx_bytes, mime_type)
        self.file_cache[path] = item
        return item.get("webUrl", path)

    def write_bytes(self, path: str, content: bytes) -> str:
        parent_dir = str(Path(path).parent)
        if parent_dir not in (".", ""):
            self._ensure_folder_path(parent_dir)

        mime_type, _ = guess_type(path)
        item = self._upload_bytes(path, content, mime_type)
        self.file_cache[path] = item
        return item.get("webUrl", path)

    def load_dataframe(self, path: str) -> Optional[pl.DataFrame]:
        mapped_path = self._map_dataframe_path(path)
        item = self._get_item_by_path(mapped_path)
        if not item:
            return None
        content = self._download_bytes(item["id"])
        if mapped_path.lower().endswith(".xlsx"):
            return pl.read_excel(io.BytesIO(content))
        return pl.read_csv(io.BytesIO(content))

    def save_dataframe(self, df: pl.DataFrame, path: str) -> str:
        mapped_path = self._map_dataframe_path(path)
        parent_dir = str(Path(mapped_path).parent)
        if parent_dir not in (".", ""):
            self._ensure_folder_path(parent_dir)

        output = io.BytesIO()
        if mapped_path.lower().endswith(".xlsx"):
            df.write_excel(output)
            content_type = (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            df.write_csv(output)
            content_type = "text/csv"
        item = self._upload_bytes(mapped_path, output.getvalue(), content_type)
        self.file_cache[path] = item
        return item.get("webUrl", path)

    def ensure_directory(self, path: str) -> None:
        if path and path not in (".", ""):
            self._ensure_folder_path(path)

    def upload_file(
        self, local_path: str, target_path: str, content_type: Optional[str] = None
    ) -> str:
        parent_dir = str(Path(target_path).parent)
        if parent_dir not in (".", ""):
            self._ensure_folder_path(parent_dir)

        if content_type is None:
            content_type, _ = guess_type(local_path)

        with open(local_path, "rb") as f:
            item = self._upload_bytes(target_path, f.read(), content_type)
        self.file_cache[target_path] = item
        return item.get("webUrl", target_path)

    def get_full_path(self, path: str) -> str:
        if path.startswith("http"):
            return path
        mapped_path = self._map_dataframe_path(self._map_text_path(path))
        item = self._get_item_by_path(mapped_path)
        if item and "webUrl" in item:
            return item["webUrl"]
        return path

    def get_local_file(
        self, path: str, download_dir: Optional[str] = None
    ) -> Optional[str]:
        if not self.exists(path):
            return None

        if not download_dir:
            download_dir = tempfile.gettempdir()

        if path.startswith("http"):
            item = self._get_item_by_url(path)
            if not item:
                return None
            filename = item.get("name", "downloaded_file")
        else:
            filename = Path(path).name

        local_path = os.path.join(download_dir, filename)
        try:
            data = self.read_bytes(path)
            with open(local_path, "wb") as f:
                f.write(data)
            return local_path
        except (OSError, requests.HTTPError) as e:
            print(f"Error downloading file {path} to local: {e}")
            return None
