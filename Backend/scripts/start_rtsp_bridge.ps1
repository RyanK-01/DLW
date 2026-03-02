param(
    [Parameter(Mandatory = $true)]
    [string]$PhoneIp,

    [int]$HttpPort = 8080,
    [int]$RtspPort = 8554,
    [string]$StreamName = "cam_01"
)

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    Write-Error "ffmpeg not found in PATH. Install ffmpeg and reopen terminal."
    exit 1
}

$inputUrl = "http://$PhoneIp`:$HttpPort/video"
$outputUrl = "rtsp://127.0.0.1`:$RtspPort/$StreamName"

Write-Host "Starting RTSP bridge..."
Write-Host "Input : $inputUrl"
Write-Host "Output: $outputUrl"

& ffmpeg `
    -fflags +genpts+discardcorrupt `
    -use_wallclock_as_timestamps 1 `
    -f mjpeg -i $inputUrl `
    -an `
    -c:v libx264 `
    -preset veryfast `
    -tune zerolatency `
    -pix_fmt yuv420p `
    -g 30 `
    -f rtsp `
    -rtsp_transport tcp `
    $outputUrl
