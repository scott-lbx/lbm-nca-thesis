# install_autostart.ps1 — 在 Windows 開機啟動資料夾建捷徑，讓 launcher.py 開機自動常駐。
# 只需跑一次。跑完後，之後每次登入都會自動起 launcher（監聽 8090），
# 開 demo/index.html 按「啟動 demo 後端」即可，不用再手動跑任何 .bat。
#
# 移除方式：刪除 shell:startup 底下的「NCA_launcher.lnk」即可（不影響其他任何東西）。

$launcherBat = Join-Path $PSScriptRoot "launcher.bat"
if (-not (Test-Path $launcherBat)) {
    Write-Host "[install_autostart] 找不到 $launcherBat，請確認路徑"
    exit 1
}

$startupDir = [Environment]::GetFolderPath("Startup")
$shortcutPath = Join-Path $startupDir "NCA_launcher.lnk"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $launcherBat
$shortcut.WorkingDirectory = Split-Path $launcherBat
$shortcut.Description = "NCA demo launcher（常駐 8090，供 demo/index.html 按鈕啟動後端）"
$shortcut.Save()

Write-Host "[install_autostart] 已建立開機啟動捷徑: $shortcutPath"
Write-Host "[install_autostart] 下次登入會自動啟動 launcher。"
Write-Host "[install_autostart] 若不想等下次登入，現在手動起一次："
Write-Host "  cmd /c `"$launcherBat`""
Write-Host "[install_autostart] 移除自動啟動：刪除 $shortcutPath"
