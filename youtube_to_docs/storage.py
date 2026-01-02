import atexit
import io
import json
import os
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
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
    """Implementation of Storage for SharePoint/OneDrive."""

    CLIENT_CONFIG_FILE = Path.home() / ".azure_client.json"
    TOKEN_CACHE_FILE = Path.home() / ".msal_token_cache.json"
    SCOPES = ["Files.ReadWrite"]

    def __init__(self, output_arg: str):
        self.token = self._get_access_token()
        self.file_cache: dict[str, dict] = {}  # path -> item metadata
        self.root_folder = "youtube-to-docs-artifacts"
        self._ensure_root_folder()

    def _get_client_config(self) -> dict:
        if not self.CLIENT_CONFIG_FILE.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {self.CLIENT_CONFIG_FILE}. "
                "Please create it with {'client_id': '...', 'authority': '...'}"
            )
        return json.loads(self.CLIENT_CONFIG_FILE.read_text(encoding="utf-8"))

    def _build_msal_app(self) -> msal.PublicClientApplication:
        config = self._get_client_config()
        client_id = config.get("client_id")
        authority = config.get("authority")

        if not client_id or not authority:
            raise ValueError(
                "Config file must contain both 'client_id' and 'authority'"
            )

        cache = msal.SerializableTokenCache()
        if self.TOKEN_CACHE_FILE.exists():
            cache.deserialize(self.TOKEN_CACHE_FILE.read_text(encoding="utf-8"))

        atexit.register(
            lambda: self.TOKEN_CACHE_FILE.write_text(
                cache.serialize(), encoding="utf-8"
            )
            if cache.has_state_changed
            else None
        )

        return msal.PublicClientApplication(
            client_id=client_id,
            authority=authority,
            token_cache=cache,
        )

    def _get_access_token(self) -> str:
        app = self._build_msal_app()
        accounts = app.get_accounts()
        result = None
        if accounts:
            result = app.acquire_token_silent(self.SCOPES, account=accounts[0])

        if not result:
            print("No cached token found, attempting interactive login...")
            result = app.acquire_token_interactive(self.SCOPES)

        if "access_token" not in result:
            error_msg = (
                result.get("error_description")
                or result.get("error")
                or "Unknown error"
            )
            raise RuntimeError(f"Could not authenticate: {error_msg}")

        return result["access_token"]

    def _ensure_root_folder(self):
        """Ensure the root folder for artifacts exists."""
        self.ensure_directory(self.root_folder)

    def _get_item_metadata(self, path: str) -> Optional[dict]:
        """Gets metadata for a file or folder."""
        if path in self.file_cache:
            return self.file_cache[path]

        # Normalize path for API
        remote_path = Path(self.root_folder, path).as_posix()
        url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{remote_path}"
        headers = {"Authorization": f"Bearer {self.token}"}
        resp = requests.get(url, headers=headers)

        if resp.status_code == 200:
            meta = resp.json()
            self.file_cache[path] = meta
            return meta
        return None

    def _extract_id_from_url(self, url: str) -> Optional[str]:
        # M365 URLs are not as straightforward to parse for IDs as Google Drive's
        # This is a placeholder and may need a more robust implementation
        # For now, we'll rely on path lookups mostly
        return None

    def exists(self, path: str) -> bool:
        if path.startswith("http"):
            return True  # Assume links exist
        return self._get_item_metadata(path) is not None

    def read_bytes(self, path: str) -> bytes:
        meta = self._get_item_metadata(path)
        if not meta or "@microsoft.graph.downloadUrl" not in meta:
            raise FileNotFoundError(f"File not found or no download URL for: {path}")

        download_url = meta["@microsoft.graph.downloadUrl"]
        resp = requests.get(download_url)
        if not resp.ok:
            raise IOError(f"Failed to download file: {path} ({resp.status_code})")
        return resp.content

    def read_text(self, path: str) -> str:
        content_bytes = self.read_bytes(path)
        if path.endswith(".docx"):
            # If it is a word doc, convert to markdown
            try:
                return pypandoc.convert_text(
                    content_bytes, to="markdown", format="docx"
                )
            except Exception as e:
                raise IOError(f"Failed to convert DOCX to text: {e}")
        return content_bytes.decode("utf-8")

    def _upload(
        self, path: str, content: bytes, content_type: str, conflict_behavior="replace"
    ) -> str:
        filename = Path(path).name
        # Get parent path relative to root
        parent_path = Path(path).parent.as_posix()
        if parent_path == ".":
            parent_path = ""
        else:
            parent_path = f"/{parent_path}"

        # URL for creating parent folder if it doesn't exist
        # and for getting the parent folder's ID
        full_parent_path = Path(self.root_folder, parent_path.strip("/")).as_posix()
        parent_url = (
            f"https://graph.microsoft.com/v1.0/me/drive/root:/{full_parent_path}"
        )
        headers = {"Authorization": f"Bearer {self.token}"}
        parent_meta = requests.get(parent_url, headers=headers).json()
        parent_id = parent_meta.get("id")

        if not parent_id:
            raise FileNotFoundError(f"Could not find or create parent folder at {parent_path}")

        upload_url = (
            f"https://graph.microsoft.com/v1.0/me/drive/items/{parent_id}:/{filename}:/content"
            f"?@microsoft.graph.conflictBehavior={conflict_behavior}"
        )
        headers["Content-Type"] = content_type

        resp = requests.put(upload_url, headers=headers, data=content)

        if not resp.ok:
            raise IOError(f"Upload failed for {path} ({resp.status_code}): {resp.text}")

        meta = resp.json()
        self.file_cache[path] = meta
        return meta.get("webUrl", "")

    def _convert_md_to_docx_bytes(self, md_text: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_docx:
            tmp_docx_path = tmp_docx.name
        try:
            pypandoc.convert_text(
                md_text,
                to="docx",
                format="md",
                outputfile=tmp_docx_path,
                extra_args=["--wrap=preserve"],
            )
            return Path(tmp_docx_path).read_bytes()
        finally:
            if os.path.exists(tmp_docx_path):
                os.remove(tmp_docx_path)

    def write_text(self, path: str, content: str) -> str:
        if path.endswith(".md"):
            # Convert markdown to Word for better viewing online
            docx_bytes = self._convert_md_to_docx_bytes(content)
            new_path = path.replace(".md", ".docx")
            return self.write_bytes(new_path, docx_bytes)
        else:
            # Assume plain text
            return self._upload(
                path, content.encode("utf-8"), "text/plain"
            )

    def write_bytes(self, path: str, content: bytes) -> str:
        filename = Path(path).name
        content_type = "application/octet-stream"
        if filename.endswith(".docx"):
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif filename.endswith(".xlsx"):
            content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif filename.endswith(".wav"):
            content_type = "audio/wav"
        elif filename.endswith(".png"):
            content_type = "image/png"

        return self._upload(path, content, content_type)

    def _convert_csv_to_xlsx_bytes(self, csv_text: str) -> bytes:
        df = pl.read_csv(io.StringIO(csv_text))
        with io.BytesIO() as output:
            df.write_excel(output)
            return output.getvalue()

    def load_dataframe(self, path: str) -> Optional[pl.DataFrame]:
        try:
            excel_bytes = self.read_bytes(path)
            return pl.read_excel(io.BytesIO(excel_bytes))
        except (FileNotFoundError, IOError, Exception) as e:
            print(f"Error loading dataframe from M365: {e}")
            return None

    def save_dataframe(self, df: pl.DataFrame, path: str) -> str:
        # Convert dataframe to xlsx in-memory
        with io.BytesIO() as xlsx_buffer:
            df.write_excel(xlsx_buffer)
            xlsx_bytes = xlsx_buffer.getvalue()

        # Upload as xlsx
        new_path = Path(path).with_suffix(".xlsx").name
        return self.write_bytes(new_path, xlsx_bytes)

    def ensure_directory(self, path: str) -> None:
        parts = Path(path).parts
        current_path = ""
        for part in parts:
            parent_path = current_path
            current_path = f"{current_path}/{part}" if current_path else part

            if self._get_item_metadata(current_path):
                continue  # Already exists

            # URL for creating folder
            if parent_path:
                parent_meta = self._get_item_metadata(parent_path)
                if not parent_meta:
                    raise IOError(f"Could not find parent folder {parent_path} for {part}")
                parent_id = parent_meta["id"]
                url = f"https://graph.microsoft.com/v1.0/me/drive/items/{parent_id}/children"
            else:
                # Top level folder
                url = "https://graph.microsoft.com/v1.0/me/drive/root/children"

            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
            body = {
                "name": part,
                "folder": {},
                "@microsoft.graph.conflictBehavior": "fail",
            }
            resp = requests.post(url, headers=headers, json=body)
            if not resp.ok and resp.status_code != 409: # 409 is conflict/exists
                 raise IOError(
                    f"Failed to create directory {part} in {parent_path} ({resp.status_code}): {resp.text}"
                )


    def upload_file(
        self, local_path: str, target_path: str, content_type: Optional[str] = None
    ) -> str:
        with open(local_path, "rb") as f:
            content = f.read()

        if content_type is None:
            import mimetypes
            content_type, _ = mimetypes.guess_type(local_path)
            if content_type is None:
                content_type = "application/octet-stream"

        return self._upload(target_path, content, content_type)

    def get_full_path(self, path: str) -> str:
        if path.startswith("http"):
            return path
        meta = self._get_item_metadata(path)
        return meta.get("webUrl", path) if meta else path

    def get_local_file(
        self, path: str, download_dir: Optional[str] = None
    ) -> Optional[str]:
        if not self.exists(path):
            return None

        if not download_dir:
            download_dir = tempfile.gettempdir()

        filename = Path(path).name
        local_path = os.path.join(download_dir, filename)

        try:
            data = self.read_bytes(path)
            with open(local_path, "wb") as f:
                f.write(data)
            return local_path
        except (IOError, OSError) as e:
            print(f"Error downloading file {path} to local: {e}")
            return None
