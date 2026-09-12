plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

// Speech on the phone, shared by Reader's Audio Player and Reader's Recorder: whisper.cpp and
// llama.cpp vendored and compiled here once, the Kotlin around them, and a small provider so that
// the two apps read one copy of each model instead of each downloading its own.
android {
    namespace = "com.freedomfighter.readers.speech"
    compileSdk = 35

    defaultConfig {
        minSdk = 26
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
        externalNativeBuild { cmake { arguments += listOf("-DGGML_NATIVE=OFF", "-DANDROID_STL=c++_static") } }
    }

    ndkVersion = "27.1.12297006"
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}
