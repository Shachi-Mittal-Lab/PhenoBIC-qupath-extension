def imageData = getCurrentImageData()
def server = imageData.getServer()

def channels = server.getMetadata().getChannels()

println "Number of channels: " + channels.size()

channels.eachWithIndex { ch, i ->
    println "Channel ${i}: " + ch.getName()
}
