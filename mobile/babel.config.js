module.exports = function(api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    // Reanimated v4 split its babel plugin out into react-native-worklets -
    // must stay last in the plugins list.
    plugins: ['react-native-worklets/plugin'],
  };
};
