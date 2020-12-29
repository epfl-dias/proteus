import EmberRouter from '@ember/routing/router';
import config from './config/environment';

const Router = EmberRouter.extend({
  location: config.locationType,
  rootURL: config.rootURL
});

Router.map(function() {
  this.route('timeline');
  this.route('plan', function() {
    this.route('show', { path: '/:plan_id' });
  });
  this.route('sqltab');
});

export default Router;
