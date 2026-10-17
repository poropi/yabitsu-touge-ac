from yabitsu import config, textures

if __name__ == "__main__":
    for name, p in textures.write_textures(config.TRACK_BUILD_DIR / "textures").items():
        print(name, p)
