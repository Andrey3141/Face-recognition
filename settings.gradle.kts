pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
        // Добавляем репозиторий для OpenCV
        maven { url = uri("https://dl.bintray.com/quickbirdstudios/OpenCV-Android") }
    }
}

rootProject.name = "Rasp"
include(":app")