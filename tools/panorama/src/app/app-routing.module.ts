import {NgModule} from '@angular/core';
import {Routes, RouterModule} from '@angular/router';
import {TimelineComponent} from './timeline/timeline.component';
import {BwProfileComponent} from './bw-profile/bw-profile.component';
import {OverviewComponent} from './overview/overview.component';
import {TimelineRangeComponent} from './timeline-range/timeline-range.component';

const routes: Routes = [
  {path: 'timeline', component: TimelineComponent},
  {path: 'timeline-range', component: TimelineRangeComponent},
  {path: 'bw-profile', component: BwProfileComponent},
  {path: 'overview', component: OverviewComponent}
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {
}
