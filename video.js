var player

function onYouTubeIframeAPIReady() {
  player = new YT.Player('player', {
    height: '390',
    width: '640',
    videoId: 'qqGhV3Flmus', // Example video ID
    playerVars: {
      autoplay: 1,
      controls: 1,
    },
    events: {
      onReady: onPlayerReady,
      onStateChange: onPlayerStateChange,
    },
  })
}

function onPlayerReady(event) {
  event.target.playVideo()
}

function onPlayerStateChange(event) {
  // You can add custom logic here for player state changes
}
