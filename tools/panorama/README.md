# Panorama

## Configuring

To configure panorama, browse with a terminal to panorama's root and run the following:
```sh
export PATH=~/.config/yarn/global/node_modules/node/bin:node_modules/.bin/:$PATH
npm install yarn
# Warning, following commands updates node **USER**-wide
yarn global add node # this is needed, probably, only if npm is too old (e.g., default npm in Ubuntu 18.04)
yarn install
```

## Moving your last timeline files into panorama
```sh
cd ${PROTEUS_INSTALL_ROOT}/opt/pelago
mv time* ${PANORAMA_ROOT}/src/assets/
```

## Development server

Run `ng serve` for a dev server. Navigate to `http://localhost:4200/`. The app will automatically reload if you change any of the source files.

## Code scaffolding

Run `ng generate component component-name` to generate a new component. You can also use `ng generate directive|pipe|service|class|guard|interface|enum|module`.

## Build

Run `ng build` to build the project. The build artifacts will be stored in the `dist/` directory. Use the `--prod` flag for a production build.

## Running unit tests

Run `ng test` to execute the unit tests via [Karma](https://karma-runner.github.io).

## Running end-to-end tests

Run `ng e2e` to execute the end-to-end tests via [Protractor](http://www.protractortest.org/).

## Further help

To get more help on the Angular CLI use `ng help` or go check out the [Angular CLI Overview and Command Reference](https://angular.io/cli) page.

### Serving panorama remotely, completely UNSAFELY and CARELESSLY
Naively running the following would serve the dev version of panorama to any host (not only localhost), without any checks.
So it's completely unsafe, use with great care!!!
```sh
ng serve --host 0.0.0.0 --disableHostCheck
```
And do not forget for any reason to terminate it as soon as possible!
