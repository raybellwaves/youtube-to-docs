# Microsoft 365 Authentication Setup

To use the Microsoft 365 storage backend, you need to register an application in Azure Active Directory and create a `~/.ms_client_secret.json` file with the application's credentials.

## 1. Register an Application in Azure AD

1.  Go to the [Azure portal](https://portal.azure.com/).
2.  Navigate to **Azure Active Directory** > **App registrations**.
3.  Click **New registration**.
4.  Give your application a name (e.g., `youtube-to-docs`).
5.  For **Supported account types**, select **Accounts in any organizational directory (Any Azure AD directory - Multitenant) and personal Microsoft accounts (e.g. Skype, Xbox)**.
6.  Leave the **Redirect URI** blank for now.
7.  Click **Register**.

## 2. Configure Authentication

1.  Once the application is registered, go to the **Authentication** tab.
2.  Click **Add a platform** and select **Mobile and desktop applications**.
3.  Check the box for `https://login.microsoftonline.com/common/oauth2/nativeclient`.
4.  Click **Configure**.
5.  Under **Advanced settings**, set **Allow public client flows** to **Yes**.
6.  Click **Save**.

## 3. Add API Permissions

1.  Go to the **API permissions** tab.
2.  Click **Add a permission**.
3.  Select **Microsoft Graph**.
4.  Select **Delegated permissions**.
5.  Search for and add the following permissions:
    *   `Files.ReadWrite.All`
    *   `User.Read`
6.  Click **Add permissions**.
7.  Click **Grant admin consent for [Your Directory Name]**.

## 4. Create the `~/.ms_client_secret.json` file

1.  Go to the **Overview** tab of your application in the Azure portal.
2.  Copy the **Application (client) ID**.
3.  Create a new file at `~/.ms_client_secret.json` with the following content:

```json
{
  "client_id": "YOUR_APPLICATION_CLIENT_ID",
  "authority": "https://login.microsoftonline.com/common"
}
```

Replace `YOUR_APPLICATION_CLIENT_ID` with the client ID you copied from the Azure portal.

## 5. First-time Authentication

The first time you run the `youtube-to-docs` command with the `-o m365` option, you will be prompted to authenticate through a device flow. Follow the on-screen instructions to complete the authentication process.
