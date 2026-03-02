param(
    [Parameter(Mandatory = $true)]
    [string]$PhoneIp,

    [int]$HttpPort = 8080,
    [int]$RtspPort = 8554,
    [string]$StreamName = "cam_01",
    [int]$Fps = 10
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

& ffmpeg 
    -f mjpeg -i "$inputUrl" 
    -vf "fps=10,scale=960:-2" 
    -an 
    -c:v libx264 
    -preset ultrafast 
    -tune zerolatency 
    -profile:v baseline 
    -level 3.1 
    -g 20 
    -x264-params "bframes=0:scenecut=0:ref=1:keyint=20:min-keyint=20" 
    -f rtsp 
    -rtsp_transport tcp 
    $outputUrl
