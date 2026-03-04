plugins {
    java
    id("com.gradleup.shadow") version "8.3.5"
    id("qupath-conventions")
}

java {
    withSourcesJar()
    withJavadocJar()
}

qupathExtension {
    name = "qupath-extension-phenobic"
    group = "io.github.qupath"
    version = "0.1.0-SNAPSHOT"
    description = "PhenoBIC cell phenotype inference: run Groovy script from Extensions menu."
    automaticModule = "io.github.qupath.extension.phenobic"
}

dependencies {
    shadow(libs.bundles.qupath)
    shadow(libs.bundles.logging)
    shadow(libs.qupath.fxtras)
    testImplementation(libs.bundles.qupath)
    testImplementation(libs.junit)
}

// Main installable JAR is the shadow (fat) JAR; build produces exactly three: main, sources, javadoc
tasks.named("build") { dependsOn("shadowJar") }
tasks.named("jar") { enabled = false }
tasks.shadowJar {
    archiveClassifier.set("")
    archiveBaseName.set("qupath-extension-phenobic")
}
